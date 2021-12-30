from copy import copy, deepcopy
from typing import Union, Sequence, Tuple, Dict, List, Optional

import dask
import dask.array as da
import numpy as np
from ase import Atoms
from dask import graph_manipulation

from abtem.core.antialias import AntialiasAperture
from abtem.core.axes import frozen_phonons_axes_metadata, HasAxesMetadata
from abtem.core.backend import get_array_module, cp, copy_to_device, _validate_device
from abtem.core.complex import complex_exponential
from abtem.core.dask import HasDaskArray, ComputableList, validate_lazy
from abtem.core.energy import Accelerator
from abtem.core.grid import Grid
from abtem.core.utils import generate_chunks
from abtem.measure.detect import AbstractDetector
from abtem.measure.measure import AbstractMeasurement
from abtem.potentials.potentials import AbstractPotential, validate_potential
from abtem.waves.base import AbstractScannedWaves
from abtem.waves.prism_utils import prism_wave_vectors, partitioned_prism_wave_vectors, plane_waves, remove_tilt, \
    interpolate_full, beamlet_basis, reduce_beamlets_nearest_no_interpolation
from abtem.waves.scan import AbstractScan, GridScan, LineScan, CustomScan
from abtem.waves.tilt import BeamTilt
from abtem.waves.transfer import CTF
from abtem.waves.waves import Waves


def wrapped_slices(start: int, stop: int, n: int) -> Tuple[slice, slice]:
    # if stop - start > n:
    # raise RuntimeError(f'{start} {stop} {n} {stop - start}')

    if start < 0:
        if stop > n:
            raise RuntimeError(f'start = {start} stop = {stop}, n = {n}')

        a = slice(start % n, None)
        b = slice(0, stop)

    elif stop > n:
        if start < 0:
            raise RuntimeError(f'start = {start} stop = {stop}, n = {n}')

        a = slice(start, None)
        b = slice(0, stop - n)

    else:
        a = slice(start, stop)
        b = slice(0, 0)
    return a, b


def wrapped_crop_2d(array: np.ndarray, corner: Tuple[int, int], size: Tuple[int, int]) -> np.ndarray:
    upper_corner = (corner[0] + size[0], corner[1] + size[1])

    xp = get_array_module(array)

    a, c = wrapped_slices(corner[0], upper_corner[0], array.shape[-2])
    b, d = wrapped_slices(corner[1], upper_corner[1], array.shape[-1])

    A = array[..., a, b]
    B = array[..., c, b]
    D = array[..., c, d]
    C = array[..., a, d]

    if A.size == 0:
        AB = B
    elif B.size == 0:
        AB = A
    else:
        AB = xp.concatenate([A, B], axis=-2)

    if C.size == 0:
        CD = D
    elif D.size == 0:
        CD = C
    else:
        CD = xp.concatenate([C, D], axis=-2)

    if CD.size == 0:
        return AB

    if AB.size == 0:
        return CD

    return xp.concatenate([AB, CD], axis=-1)


def batch_crop_2d(array: np.ndarray, corners: Tuple[int, int], new_shape: Tuple[int, int]):
    xp = get_array_module(array)
    if xp is cp:
        i = xp.arange(array.shape[0])[:, None, None]
        ix = cp.arange(new_shape[0]) + cp.asarray(corners[:, 0, None])
        iy = cp.arange(new_shape[1]) + cp.asarray(corners[:, 1, None])
        ix = ix[:, :, None]
        iy = iy[:, None]
        return array[i, ix, iy]
    else:
        array = np.lib.stride_tricks.sliding_window_view(array, (1,) + new_shape)
        return array[xp.arange(array.shape[0]), corners[:, 0], corners[:, 1], 0]


_plane_waves_axes_metadata = {'label': 'plane_waves', 'type': 'ensemble'}


def prism_coefficients(positions: np.ndarray, wave_vectors: np.ndarray, wavelength: float, ctf: CTF):
    xp = get_array_module(wave_vectors)
    positions = copy_to_device(positions, xp)

    def calculate_ctf_coefficient(wave_vectors, wavelength, ctf):
        alpha = xp.sqrt(wave_vectors[:, 0] ** 2 + wave_vectors[:, 1] ** 2) * wavelength
        phi = xp.arctan2(wave_vectors[:, 0], wave_vectors[:, 1])
        coefficients = ctf.evaluate(alpha, phi)
        return coefficients

    def calculate_translation_coefficients(wave_vectors, positions):
        coefficients = complex_exponential(-2. * xp.pi * positions[..., 0, None] * wave_vectors[:, 0][None])
        coefficients *= complex_exponential(-2. * xp.pi * positions[..., 1, None] * wave_vectors[:, 1][None])
        return coefficients

    return calculate_ctf_coefficient(wave_vectors, wavelength=wavelength, ctf=ctf) * \
           calculate_translation_coefficients(wave_vectors, positions)


def _minimum_crop(positions: Union[Sequence[float], AbstractScan], sampling, shape):
    if isinstance(positions, AbstractScan):
        positions = positions.limits

    offset = (shape[0] // 2, shape[1] // 2)
    corners = np.rint(np.array(positions) / sampling - offset).astype(np.int)
    upper_corners = corners + np.asarray(shape)

    crop_corner = (np.min(corners[..., 0]).item(), np.min(corners[..., 1]).item())

    size = (np.max(upper_corners[..., 0]).item() - crop_corner[0],
            np.max(upper_corners[..., 1]).item() - crop_corner[1])

    corners -= crop_corner
    return crop_corner, size, corners


def _validate_interpolation(interpolation):
    if isinstance(interpolation, int):
        interpolation = (interpolation,) * 2
    elif not len(interpolation) == 2:
        raise ValueError('Interpolation factor must be int')
    return interpolation


def _flatten(l):
    for el in l:
        if isinstance(el, list) and not isinstance(el, (str, bytes)):
            yield from _flatten(el)
        else:
            yield el


def _split_list(lst, division):
    return [lst[i:i + division] for i in range(0, len(lst), division)]


def _concatenate_blocks(list_of_arrays, block_shape):
    if len(block_shape) == 1:
        return np.concatenate(list_of_arrays, axis=0)
    elif len(block_shape) == 2:
        assert len(list_of_arrays) == block_shape[0] * block_shape[1]
        measurement = _split_list(list_of_arrays, block_shape[1])
        return np.concatenate([np.concatenate(block, axis=1) for block in measurement], axis=0)


def _shape(lst, shape=()):
    if not isinstance(lst, list):
        return shape

    if isinstance(lst[0], list):
        if not all(len(item) == len(lst[0]) for item in lst):
            msg = 'not all lists have the same length'
            raise ValueError(msg)

    shape += (len(lst),)
    shape = _shape(lst[0], shape)
    return shape


class SMatrixArray(HasDaskArray, HasAxesMetadata, AbstractScannedWaves):
    """
    Scattering matrix array object.

    The scattering matrix array object represents a plane wave expansion of a probe, it is used for STEM simulations
    with the PRISM algorithm.

    Parameters
    ----------
    array : 3d array or 4d array
        The array representation of the scattering matrix.
    energy : float
        Electron energy [eV].
    wave_vectors : 2d array
        The spatial frequencies of each plane in the plane wave expansion.
    planewave_cutoff : float
        The angular cutoff of the plane wave expansion [mrad].
    interpolation : int or two int
        Interpolation factor. Default is 1 (no interpolation).
    sampling : one or two float, optional
        Lateral sampling of wave functions [1 / Å]. Default is None (inherits sampling from the potential).
    tilt : two float, optional
        Small angle beam tilt [mrad].
    antialias_aperture : two float, optional
        Assumed antialiasing aperture as a fraction of the real space Nyquist frequency. Default is 2/3.
    device : str, optional
        The calculations will be carried out on this device. Default is 'cpu'.
    extra_axes_metadata : list of dicts
    metadata : dict
    """

    def __init__(self,
                 array: Union[np.ndarray, da.core.Array],
                 energy: float,
                 wave_vectors: np.ndarray,
                 planewave_cutoff: float,
                 interpolation: Union[int, Tuple[int, int]] = 1,
                 partitions: int = None,
                 sampling: Union[float, Tuple[float, float]] = None,
                 tilt: Tuple[float, float] = None,
                 accumulated_defocus: float = 0.,
                 crop_offset: Tuple[int, int] = (0, 0),
                 uncropped_gpts: Tuple[int, int] = None,
                 antialias_aperture: float = None,
                 device: str = None,
                 extra_axes_metadata: List[Dict] = None,
                 metadata: Dict = None):

        self._interpolation = _validate_interpolation(interpolation)

        self._grid = Grid(gpts=array.shape[-2:], sampling=sampling, lock_gpts=True)

        self._beam_tilt = BeamTilt(tilt=tilt)
        self._antialias_aperture = AntialiasAperture(cutoff=antialias_aperture)
        self._accelerator = Accelerator(energy=energy)
        self._device = _validate_device(device)

        self._array = array
        self._wave_vectors = wave_vectors
        self._planewave_cutoff = planewave_cutoff

        super().__init__(array)

        self._extra_axes_metadata = self._validate_extra_axes_metadata(extra_axes_metadata)
        self._metadata = metadata

        self._accumulated_defocus = accumulated_defocus
        self._partitions = partitions

        self._crop_offset = crop_offset
        self._uncropped_gpts = uncropped_gpts

    def __len__(self) -> int:
        return len(self.wave_vectors)

    @property
    def full_wave_vectors(self):
        return prism_wave_vectors(self.planewave_cutoff, self.extent, self.energy, self.interpolation)

    @property
    def crop_offset(self) -> Tuple[int, int]:
        return self._crop_offset

    @property
    def uncropped_gpts(self) -> Tuple[int, int]:
        if self._uncropped_gpts is None:
            return self.gpts
        return self._uncropped_gpts

    @property
    def is_cropped(self) -> bool:
        return self.uncropped_gpts != self.gpts

    @property
    def planewave_cutoff(self) -> float:
        return self._planewave_cutoff

    @property
    def num_axes(self) -> int:
        return len(self.array.shape)

    @property
    def num_base_axes(self) -> int:
        return 3

    @property
    def chunks(self) -> int:
        return self.array.chunks[:-2]

    @property
    def axes_metadata(self) -> List[Dict]:
        return self._extra_axes_metadata + [_plane_waves_axes_metadata] + self._base_axes_metadata

    @property
    def metadata(self) -> Dict:
        return self._metadata

    @property
    def accumulated_defocus(self) -> float:
        return self._accumulated_defocus

    @accumulated_defocus.setter
    def accumulated_defocus(self, value):
        self._accumulated_defocus = value

    @property
    def wave_vectors(self) -> np.ndarray:
        """The spatial frequencies of each wave in the plane wave expansion."""
        return self._wave_vectors

    @property
    def interpolation(self) -> Tuple[int, int]:
        """Interpolation factor."""
        return self._interpolation

    @property
    def partitions(self) -> int:
        return self._partitions

    @property
    def interpolated_gpts(self) -> Tuple[int, int]:
        return self.uncropped_gpts[0] // self.interpolation[0], self.uncropped_gpts[1] // self.interpolation[1]

    def rechunk(self, chunks: int = None, **kwargs):
        if not isinstance(self.array, da.core.Array):
            raise RuntimeError()

        if chunks is None:
            chunks = self.array.chunks[:-3] + ((sum(self.array.chunks[-3]),),) + self.array.chunks[-2:]

        self._array = self._array.rechunk(chunks=chunks, **kwargs)
        return self

    def crop_to_positions(self, positions: Union[np.ndarray, AbstractScan]):
        xp = get_array_module(self.array)
        if self.interpolation == (1, 1):
            corner = (0, 0)
            cropped_array = self.array
        else:
            corner, size, _ = _minimum_crop(positions, self.sampling, self.interpolated_gpts)
            corner = (corner[0] if self.interpolation[0] > 1 else 0, corner[1] if self.interpolation[1] > 1 else 0)

            size = (size[0] if self.interpolation[0] > 1 else self.gpts[0],
                    size[1] if self.interpolation[1] > 1 else self.gpts[1])

            if self.is_lazy:
                cropped_array = self.array.map_blocks(wrapped_crop_2d,
                                                      corner=corner,
                                                      size=size,
                                                      chunks=self.array.chunks[:-2] + ((size[0],), (size[1],)),
                                                      meta=xp.array((), dtype=xp.complex64))
            else:
                cropped_array = wrapped_crop_2d(self.array, corner=corner, size=size)

        d = self._copy_as_dict(copy_array=False)
        d['array'] = cropped_array
        d['crop_offset'] = corner
        d['uncropped_gpts'] = self.uncropped_gpts
        return self.__class__(**d)

    def downsample(self, max_angle: Union[str, float] = 'cutoff', normalization: str = 'values') -> 'SMatrixArray':
        waves = Waves(self.array, sampling=self.sampling, energy=self.energy)
        waves = waves.downsample(max_angle=max_angle, normalization=normalization)
        d = self._copy_as_dict(copy_array=False)
        d['array'] = waves.array
        d['sampling'] = waves.sampling
        return self.__class__(**d)

    def multislice(self, potential: AbstractPotential) -> 'SMatrixArray':
        """
        Propagate the scattering matrix through the provided potential.

        Parameters
        ----------
        potential : AbstractPotential object
            Scattering potential.

        Returns
        -------
        Waves object.
            Probe exit wave functions for the provided positions.
        """
        array = Waves(self.array, energy=self.energy, sampling=self.sampling).multislice(potential).array
        d = self._copy_as_dict(copy_array=False)
        d['array'] = array
        return self.__class__(**d)

    def remove_tilt(self):
        xp = get_array_module(self.array)
        if self.is_lazy:
            array = self.array.map_blocks(remove_tilt,
                                          planewave_cutoff=self.planewave_cutoff,
                                          extent=self.extent,
                                          gpts=self.gpts,
                                          energy=self.energy,
                                          interpolation=self.interpolation,
                                          partitions=self.partitions,
                                          accumulated_defocus=self.accumulated_defocus,
                                          meta=xp.array((), dtype=xp.complex64))
        else:
            array = remove_tilt(self.array,
                                planewave_cutoff=self.planewave_cutoff,
                                extent=self.extent,
                                gpts=self.gpts,
                                energy=self.energy,
                                interpolation=self.interpolation,
                                partitions=self.partitions,
                                accumulated_defocus=self.accumulated_defocus)

        self._array = array
        return self

    def interpolate_full(self, chunks):
        xp = get_array_module(self.array)
        self.remove_tilt()
        self.rechunk()

        wave_vectors = prism_wave_vectors(self.planewave_cutoff, self.extent, self.energy, self.interpolation)

        arrays = []
        for start, end in generate_chunks(len(wave_vectors), chunks=chunks):
            array = dask.delayed(interpolate_full)(array=self.array,
                                                   parent_wave_vectors=self.wave_vectors,
                                                   wave_vectors=wave_vectors[start:end],
                                                   extent=self.extent,
                                                   gpts=self.gpts,
                                                   energy=self.energy,
                                                   defocus=self.accumulated_defocus)

            array = da.from_delayed(array, shape=(end - start,) + self.gpts, dtype=xp.complex64)
            array = array * np.sqrt(len(self) / len(wave_vectors))
            arrays.append(array)

        array = da.concatenate(arrays)
        d = self._copy_as_dict(copy_array=False)
        d['array'] = array
        d['wave_vectors'] = wave_vectors
        return self.__class__(**d)

    def _validate_positions_per_reduction(self, positions_per_reduction):
        if positions_per_reduction == 'auto' or positions_per_reduction is None:
            positions_per_reduction = 300
        return positions_per_reduction

    def _validate_ctf(self, ctf):
        if ctf is None:
            ctf = CTF(semiangle_cutoff=self.planewave_cutoff, energy=self.energy)

        if isinstance(ctf, dict):
            ctf = CTF(energy=self.energy, **ctf)

        return ctf

    def _validate_positions(self, positions, default_sampling: Tuple[float, float]):
        if positions is None:
            positions = (0., 0.)

        if isinstance(positions, GridScan):
            if positions.start is None:
                positions.start = (0., 0.)
            if positions.end is None:
                positions.end = self.extent
            if positions.sampling is None:
                positions.sampling = default_sampling
            return positions

        elif isinstance(positions, LineScan):
            raise NotImplementedError()

        elif isinstance(positions, (list, tuple, np.ndarray)):
            return CustomScan(positions)

        else:
            raise NotImplementedError

    def _get_s_matrices(self):

        if len(self.array.shape) == 3:
            return [self]

        s_matrices = []
        for i in range(len(self.array)):
            d = self._copy_as_dict(copy_array=False)
            d['array'] = self.array[i]
            d['extra_axes_metadata'] = []
            s_matrices.append(self.__class__(**d))

        return s_matrices

    def _reduce_partitioned(self, array: np.ndarray, positions: np.ndarray, basis) -> Waves:
        shifts = np.round(positions.reshape((-1, 2)) / self.sampling).astype(int)
        shifts -= np.array(self.crop_offset)
        shifts -= (np.array(self.interpolated_gpts)) // 2

        basis = np.moveaxis(basis, 0, 2).copy()
        array = np.moveaxis(array, 0, 2).copy()

        waves = np.zeros((len(shifts),) + self.interpolated_gpts, dtype=np.complex64)

        reduce_beamlets_nearest_no_interpolation(waves, basis, array, shifts)

        waves = waves.reshape(positions.shape[:-1] + waves.shape[-2:])

        extra_axes_metadata = [{'type': 'positions'} for _ in range(len(positions.shape[:-1]))]

        waves = Waves(waves,
                      sampling=self.sampling,
                      energy=self.energy,
                      extra_axes_metadata=extra_axes_metadata,
                      antialias_aperture=self.antialias_aperture)
        return waves

    def _reduce(self, array: np.ndarray, positions: np.ndarray, ctf: CTF) -> Waves:
        xp = get_array_module(array)
        offset_positions = positions - np.array(self.crop_offset) * np.array(self.sampling)
        coefficients = prism_coefficients(positions.reshape((-1, 2)), self.wave_vectors, self.wavelength, ctf)

        if not array.shape[-2:] == self.interpolated_gpts:
            crop_corner, size, corners = _minimum_crop(offset_positions, self.sampling, self.interpolated_gpts)

            array = wrapped_crop_2d(array, crop_corner, size)
            window = xp.tensordot(coefficients, array, axes=[-1, -3])
            window = batch_crop_2d(window, corners.reshape((-1, 2)), self.interpolated_gpts)
        else:
            window = xp.tensordot(coefficients, array, axes=[-1, -3])

        window = window.reshape(offset_positions.shape[:-1] + window.shape[-2:])

        extra_axes_metadata = [{'type': 'positions'} for _ in range(len(offset_positions.shape) - 1)]

        waves = Waves(window,
                      sampling=self.sampling,
                      energy=self.energy,
                      extra_axes_metadata=extra_axes_metadata,
                      antialias_aperture=self.antialias_aperture)
        return waves

    def _launch_reductions(self, array, detectors, scan, ctf, positions_per_reduction):
        position_blocks = scan.get_positions(chunks=positions_per_reduction, lazy=False)

        kwargs = {}
        if self.partitions:
            extent = (self.interpolated_gpts[0] * self.sampling[0], self.interpolated_gpts[1] * self.sampling[1])
            wave_vectors = prism_wave_vectors(self.planewave_cutoff, extent, self.energy, (1, 1))
            ctf = ctf.copy()
            ctf.defocus = -self.accumulated_defocus
            basis = beamlet_basis(ctf,
                                  self.wave_vectors,
                                  wave_vectors,
                                  self.interpolated_gpts,
                                  self.sampling).astype(np.complex64)
            basis = np.fft.fftshift(basis, axes=(1, 2))
            kwargs['basis'] = basis
        else:
            kwargs['ctf'] = ctf

        # import matplotlib.pyplot as plt
        #
        # # print(self.uncropped_gpts, self.interpolated_gpts)
        #
        # basis = kwargs['basis']
        # basis = basis[:, :self.interpolated_gpts[0], :self.interpolated_gpts[1]]
        #
        # plt.imshow(np.fft.fftshift(basis.sum(0).real))
        # plt.show()
        #
        # sss
        measurements = [[] for _ in range(len(detectors))]
        for positions in _flatten(position_blocks):

            if self.partitions is None:
                waves = self._reduce(array, positions, **kwargs)
            else:
                waves = self._reduce_partitioned(array, positions, **kwargs)

            for i, detector in enumerate(detectors):
                measurements[i].append(detector.detect(waves).array)

        measurements = [_concatenate_blocks(measurement, _shape(position_blocks)) for measurement in measurements]
        return measurements

    def _distribute_reductions(self, detectors, scan, scan_partitions, ctf, positions_per_reduction):

        if scan_partitions is None:
            scan_partitions = np.prod(self.interpolation)

        if scan_partitions == 1:
            scans = [scan]
        else:
            scans = scan.divide(scan_partitions)

        distributed_measurements = []
        for scan in _flatten(scans):
            if scan_partitions == 1:
                cropped_s_matrix_array = self
            else:
                cropped_s_matrix_array = self.crop_to_positions(scan)

            if self.is_lazy:
                measurements = dask.delayed(cropped_s_matrix_array._launch_reductions, nout=len(detectors))(
                    cropped_s_matrix_array.array,
                    detectors, scan, ctf,
                    positions_per_reduction
                )

                measurements = list(measurements)
                for i, (measurement, detector) in enumerate(zip(measurements, detectors)):
                    measurements[i] = da.from_delayed(measurement,
                                                      shape=scan.shape + detector.detected_shape(self),
                                                      meta=np.array((), dtype=np.float32))


            else:
                measurements = cropped_s_matrix_array._launch_reductions(cropped_s_matrix_array.array,
                                                                         detectors,
                                                                         scan,
                                                                         ctf,
                                                                         positions_per_reduction)

            distributed_measurements.append(measurements)

        measurements = list(map(list, zip(*distributed_measurements)))

        for i, measurement in enumerate(measurements):
            measurements[i] = _concatenate_blocks(measurement, _shape(scans))

        return measurements

    def reduce(self,
               detectors: Union[AbstractDetector, List[AbstractDetector]] = None,
               positions: Union[np.ndarray, AbstractScan] = None,
               ctf: Union[CTF, Dict] = None,
               scan_partitions: int = None,
               positions_per_reduction: int = None,
               max_concurrent: int = None) -> Union[Waves, AbstractMeasurement, List[AbstractMeasurement]]:

        """
        Scan the probe across the potential and record a measurement for each detector.

        Parameters
        ----------
        detectors : List of Detector objects
            The detectors recording the measurements.
        positions : Scan object
            Scan defining the positions of the probe wave functions.
        ctf: CTF object, optional
            The probe contrast transfer function. Default is None (aperture is set by the planewave cutoff).
        scan_partitions : two int, optional
            Partitioning of the scan. The scattering matrix will be reduced in similarly partitioned chunks.
            Should be equal to or greater than the interpolation.
        positions_per_reduction : int, optional
            Number of positions per reduction operation.
        max_concurrent : int, optional
            Maximum number of scattering matrices in memory at any point.
        """
        positions_per_reduction = self._validate_positions_per_reduction(positions_per_reduction)

        detectors = self._validate_detectors(detectors)

        ctf = self._validate_ctf(ctf)

        scan = self._validate_positions(positions, ctf.nyquist_sampling)

        measurements = []
        for i, s_matrix in enumerate(self._get_s_matrices()):
            measurement = s_matrix._distribute_reductions(detectors, scan, scan_partitions, ctf,
                                                          positions_per_reduction)

            if max_concurrent is not None:
                if i >= max_concurrent:
                    measurement = graph_manipulation.bind(measurement, measurements[i - max_concurrent])
            else:
                measurement = graph_manipulation.wait_on(measurement)

            measurements.append(measurement)

        measurements = list(map(np.stack, map(list, zip(*measurements))))

        for i, (detector, measurement) in enumerate(zip(detectors, measurements)):
            extra_axes_metadata = [{'label': 'frozen_phonons', 'type': 'ensemble'}]
            measurements[i] = detector.measurement_from_array(measurement, waves=self, scan=scan,
                                                              extra_axes_metadata=extra_axes_metadata)

            if detector.ensemble_mean:
                measurements[i] = measurements[i].mean(0)

        if len(measurements) == 1:
            return measurements[0]
        else:
            return ComputableList(measurements)

    def _copy_as_dict(self, copy_array: bool = True):
        d = {'energy': self.energy,
             'wave_vectors': self.wave_vectors.copy(),
             'interpolation': self.interpolation,
             'planewave_cutoff': self.planewave_cutoff,
             'sampling': self.sampling,
             'accumulated_defocus': self.accumulated_defocus,
             'crop_offset': self.crop_offset,
             'uncropped_gpts': self._uncropped_gpts,
             'tilt': self.tilt,
             'partitions': self.partitions,
             'antialias_aperture': self.antialias_aperture,
             'device': self._device,
             'extra_axes_metadata': deepcopy(self._extra_axes_metadata),
             'metadata': copy(self.metadata)}

        if copy_array:
            d['array'] = self.array.copy()
        return d

    def copy(self):
        """Make a copy."""
        return self.__class__(**self._copy_as_dict())


class SMatrix(AbstractScannedWaves):
    """
    Scattering matrix builder class

    The scattering matrix builder object is used for creating scattering matrices and simulating STEM experiments using
    the PRISM algorithm.

    Parameters
    ----------
    potential : Atoms or Potential
    energy : float
        Electron energy [eV].
    planewave_cutoff : float
        The angular cutoff of the plane wave expansion [mrad].
    interpolation : one or two int, optional
        Interpolation factor. Default is 1 (no interpolation).
    extent : one or two float, optional
        Lateral extent of wave functions [Å]. Default is None (inherits the extent from the potential).
    gpts : one or two int, optional
        Number of grid points describing the wave functions. Default is None (inherits the gpts from the potential).
    sampling : one or two float, None
        Lateral sampling of wave functions [1 / Å]. Default is None (inherits the sampling from the potential).
    chunks :
    tilt : two float
        Small angle beam tilt [mrad].
    device : str, optional
        The calculations will be carried out on this device. Default is 'cpu'.
    """

    def __init__(self,
                 potential: Union[Atoms, AbstractPotential] = None,
                 energy: float = None,
                 planewave_cutoff: float = None,
                 interpolation: Union[int, Tuple[int, int]] = 1,
                 partitions: int = None,
                 normalization: str = None,
                 extent: Union[float, Tuple[float, float]] = None,
                 gpts: Union[int, Tuple[int, int]] = None,
                 sampling: Union[float, Tuple[float, float]] = None,
                 chunks: int = 'auto',
                 tilt: Tuple[float, float] = None,
                 device: str = None):

        self._potential = validate_potential(potential)

        if potential is None:
            self._grid = Grid(extent=extent, gpts=gpts, sampling=sampling)
        else:
            self._grid = potential.grid

        self._interpolation = _validate_interpolation(interpolation)
        self._planewave_cutoff = planewave_cutoff

        self._accelerator = Accelerator(energy=energy)
        self._beam_tilt = BeamTilt(tilt=tilt)
        self._antialias_aperture = AntialiasAperture()
        self._partitions = partitions

        self._device = _validate_device(device)

        if normalization is None and partitions:
            normalization = 'planewaves'
        elif normalization is None:
            normalization = 'probe'

        self._normalization = normalization
        self._chunks = chunks

    @property
    def potential(self) -> AbstractPotential:
        return self._potential

    @property
    def axes_metadata(self) -> List[Dict]:
        return [_plane_waves_axes_metadata] + self._base_axes_metadata

    @property
    def chunks(self) -> int:
        return self._chunks

    @property
    def normalization(self):
        return self._normalization

    @property
    def planewave_cutoff(self) -> float:
        """Plane wave expansion cutoff."""
        return self._planewave_cutoff

    @planewave_cutoff.setter
    def planewave_cutoff(self, value: float):
        self._planewave_cutoff = value

    @property
    def interpolation(self) -> Tuple[int, int]:
        """Interpolation factor."""
        return self._interpolation

    @property
    def interpolated_gpts(self) -> Tuple[int, int]:
        return self.gpts[0] // self.interpolation, self.gpts[1] // self.interpolation

    @property
    def partitions(self):
        return self._partitions

    def _get_s_matrices(self, lazy: bool = None) -> List['SMatrix']:
        lazy = validate_lazy(lazy)

        s_matrices = []

        def _s_matrix_configuration(potential):
            d = self._copy_as_dict(copy_potential=False)
            d['potential'] = potential
            return self.__class__(**d)

        if self.potential is None:
            return [_s_matrix_configuration(None)]

        for potential in self.potential.get_potential_configurations(lazy=lazy):

            if lazy:
                s_matrix = dask.delayed(_s_matrix_configuration)(potential)
            else:
                s_matrix = _s_matrix_configuration(potential)

            s_matrices.append(s_matrix)

        return s_matrices

    def _build_chunk(self, start: int = 0, end: int = None) -> np.ndarray:
        if self.partitions is None:
            wave_vectors = prism_wave_vectors(self.planewave_cutoff, self.extent, self.energy, self.interpolation)

        else:
            wave_vectors = partitioned_prism_wave_vectors(self.planewave_cutoff, self.extent, self.energy,
                                                          self.partitions)

        if end is None:
            end = len(wave_vectors)

        array = plane_waves(wave_vectors[start:end], self.extent, self.gpts)

        if self._normalization == 'probe':
            normalization_constant = np.prod(self.gpts) * np.sqrt(len(wave_vectors)) / np.prod(self.interpolation)
            array = array / normalization_constant.astype(np.float32)

        if self.potential is not None:
            waves = Waves(array, extent=self.extent, energy=self.energy)
            return waves.multislice(self.potential).array
        else:
            return array

    def build(self, lazy: bool = None) -> SMatrixArray:
        """
        Build scattering matrix and propagate the scattering matrix through the provided potential.

        Parameters
        ----------
        lazy : bool

        Returns
        -------
        SMatrixArray object
        """
        self.grid.check_is_defined()

        xp = get_array_module(self._device)
        lazy = validate_lazy(lazy)

        def _build_chunk(s_matrix, start, end):
            return s_matrix._build_chunk(start, end)

        s_matrix_arrays = []
        for s_matrix in self._get_s_matrices(lazy):
            s_matrix_array = []
            for start, end in generate_chunks(len(self), chunks=self._validate_chunks(self.chunks)):
                if lazy:
                    array = dask.delayed(_build_chunk)(s_matrix, start, end)
                    array = da.from_delayed(array, shape=(end - start,) + self.gpts,
                                            meta=xp.array((), dtype=xp.complex64))
                else:
                    array = _build_chunk(s_matrix, start, end)

                s_matrix_array.append(array)

            s_matrix_array = xp.concatenate(s_matrix_array)
            s_matrix_arrays.append(s_matrix_array)

        if len(s_matrix_arrays) > 1:
            s_matrix_arrays = np.stack(s_matrix_arrays)
            extra_axes_metadata = [frozen_phonons_axes_metadata]
        else:
            s_matrix_arrays = s_matrix_arrays[0]
            extra_axes_metadata = []

        s_matrix_array = SMatrixArray(s_matrix_arrays,
                                      interpolation=self.interpolation,
                                      planewave_cutoff=self.planewave_cutoff,
                                      sampling=self.sampling,
                                      energy=self.energy,
                                      tilt=self.tilt,
                                      partitions=self.partitions,
                                      wave_vectors=self.wave_vectors,
                                      antialias_aperture=self.antialias_aperture,
                                      device=self._device,
                                      extra_axes_metadata=extra_axes_metadata)

        if self.potential is not None:
            s_matrix_array.accumulated_defocus = self.potential.thickness

        return s_matrix_array

    def reduce(self,
               detectors: Union[AbstractDetector, List[AbstractDetector]] = None,
               positions: Union[np.ndarray, AbstractScan] = None,
               ctf: Union[CTF, Dict] = None,
               scan_partitions: Tuple[int, int] = None,
               positions_per_reduction: int = None,
               max_concurrent: int = None,
               lazy: bool = None) -> Union[Waves, AbstractMeasurement, List[AbstractMeasurement]]:

        """
        Scan the probe across the potential and record a measurement for each detector.

        Parameters
        ----------
        detectors : List of Detector objects
            The detectors recording the measurements.
        positions : Scan object
            Scan defining the positions of the probe wave functions.
        ctf: CTF object, optional
            The probe contrast transfer function. Default is None (aperture is set by the planewave cutoff).
        scan_partitions : two int, optional
            Partitioning of the scan. The scattering matrix will be reduced in similarly partitioned chunks.
            Should be equal to or greater than the interpolation.
        positions_per_reduction : int, optional
            Number of positions per reduction operation.
        max_concurrent : int, optional
            Maximum number of scattering matrices in memory at any point.
        """

        lazy = validate_lazy(lazy)

        s = self.build(lazy=lazy)

        if self.normalization == 'probe':
            s = s.downsample(normalization='amplitude')
        elif self.normalization == 'planewaves':
            s = s.downsample(normalization='values')
        else:
            raise RuntimeError()

        if self.partitions:
            s.remove_tilt()

        measurements = s.reduce(detectors=detectors,
                                positions=positions,
                                ctf=ctf,
                                scan_partitions=scan_partitions,
                                positions_per_reduction=positions_per_reduction,
                                max_concurrent=max_concurrent)
        return measurements

    def __len__(self) -> int:
        return len(self.wave_vectors)

    @property
    def wave_vectors(self) -> np.ndarray:
        self.grid.check_is_defined()
        self.accelerator.check_is_defined()

        if self.partitions is None:
            wave_vectors = prism_wave_vectors(self.planewave_cutoff, self.extent, self.energy, self.interpolation)

        else:
            wave_vectors = partitioned_prism_wave_vectors(self.planewave_cutoff, self.extent, self.energy,
                                                          num_rings=self.partitions)
        return wave_vectors

    def _copy_as_dict(self, copy_potential: bool = True):
        potential = self.potential
        if copy_potential and self.potential is not None:
            potential = potential.copy()
        d = {'potential': potential,
             'energy': self.energy,
             'planewave_cutoff': self.planewave_cutoff,
             'interpolation': self.interpolation,
             'partitions': self.partitions,
             'normalization': self._normalization,
             'extent': self.extent,
             'gpts': self.gpts,
             'sampling': self.sampling,
             'chunks': self.chunks,
             'tilt': self.tilt,
             'device': self._device}
        return d

    def __copy__(self) -> 'SMatrix':
        return self.__class__(**self._copy_as_dict())

    def copy(self) -> 'SMatrix':
        """Make a copy."""
        return copy(self)
