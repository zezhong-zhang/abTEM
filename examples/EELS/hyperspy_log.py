#!/usr/bin/env python 
# ============================
# 2021-10-01 
# 18:02 
# ============================
from abtem.visualize.utils import domain_coloring
def plot_colors(
                array,
                # extent,
                add_colorbars: bool = True,
                num_abs = 5
                ):
    from cplot._colors import get_srgb1,scale01
    from matplotlib import colors,cm
    plt.imshow(
        domain_coloring(array),
        # extent=extent,
        interpolation="nearest",
        origin="lower",
        aspect="equal",
    )

    if add_colorbars:
        # abs colorbar
        norm = colors.Normalize(vmin=np.min(np.abs(array)), vmax=np.max(np.abs(array)))
        cb0 = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cm.gray))
        cb0.set_label("abs", rotation=0, ha="center", va="top")
        cb0.ax.yaxis.set_label_coords(0.5, -0.03)
        abs_sampling = np.linspace(np.min(np.abs(array)),np.max(np.abs(array)),num=num_abs)
        str_abs_sampling = ["{:.2E}".format(n) for n in abs_sampling]
        scaled_vals = scale01(abs_sampling,abs_scaling)
        cb0.set_ticks([0.0, *scaled_vals, 1.0])
        cb0.set_ticklabels(str_abs_sampling)

        # arg colorbar
        # create new colormap
        z = np.exp(1j * np.linspace(-np.pi, np.pi, 256))
        rgb_vals = get_srgb1(z, abs_scaling=abs_scaling, colorspace=colorspace)
        rgba_vals = np.pad(rgb_vals, ((0, 0), (0, 1)), constant_values=1.0)
        newcmp = colors.ListedColormap(rgba_vals)
        #
        norm = colors.Normalize(vmin=-np.pi, vmax=np.pi)
        cb1 = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=newcmp))
        cb1.set_label("arg", rotation=0, ha="center", va="top")
        cb1.ax.yaxis.set_label_coords(0.5, -0.03)
        cb1.set_ticks([-np.pi, -np.pi / 2, 0, +np.pi / 2, np.pi])
        cb1.set_ticklabels(["-π", "-π/2", "0", "π/2", "π"])

from pyxem.utils.pixelated_stem_tools import _get_rgb_phase_magnitude_array
def make_color_wheel(ax, rotation=None):
    x, y = np.mgrid[-2.0:2.0:500j, -2.0:2.0:500j]
    r = (x ** 2 + y ** 2) ** 0.5
    t = np.arctan2(x, y)
    del x, y
    if rotation is not None:
        t += math.radians(rotation)
        t = (t + np.pi) % (2 * np.pi) - np.pi

    r_masked = np.ma.masked_where((2.0 < r) | (r < 1.0), r)
    r_masked -= 1.0

    mask = r_masked.mask
    r_masked.data[r_masked.mask] = r_masked.mean()
    rgb_array = _get_rgb_phase_magnitude_array(t, r_masked.data)
    rgb_array = np.dstack((rgb_array, np.invert(mask)))

    ax.imshow(rgb_array, interpolation="quadric", origin="lower")
    ax.set_axis_off()
ax_indicator = fig.add_subplot(331)
make_color_wheel(ax_indicator)
ax_indicator = fig.add_subplot(331)
make_color_wheel(ax_indicator)
plt.show()
ax_indicator = plt.subplot(331)
make_color_wheel(ax_indicator)
plt.show()
ax_indicator = plt.subplot(331)
make_color_wheel(ax_indicator,rotation=30)
plt.show()
ax_indicator = plt.subplot(331)
make_color_wheel(ax_indicator)
plt.show()
ax_indicator = plt.add_subplot(331)
make_color_wheel(ax_indicator)
plt.show()
ax_indicator = fig.add_subplot(331)
make_color_wheel(ax_indicator)
plt.show()
fig = ax.figure
ax_indicator = fig.add_subplot(331)
make_color_wheel(ax_indicator)
plt.show()
fig, ax = plt.subplots(1, 1, figsize=(7, 7))
ax_indicator = fig.add_subplot(331)
make_color_wheel(ax_indicator)
plt.show()
fig, ax = plt.subplots(1, 1, figsize=(7, 7))
ax_indicator = fig.add_subplot(332)
make_color_wheel(ax_indicator)
plt.show()
fig, ax = plt.subplots(1, 1, figsize=(7, 7))
ax_indicator = fig.add_subplot(331)
make_color_wheel(ax_indicator)
plt.show()
from abtem.visualize.utils import domain_coloring
def plot_colors(
                array,
                # extent,
                add_colorbars: bool = True,
                num_abs = 5
                ):
    from cplot._colors import get_srgb1,scale01
    from matplotlib import colors,cm
    plt.imshow(
        domain_coloring(array),
        # extent=extent,
        interpolation="nearest",
        origin="lower",
        aspect="equal",
    )

    if add_colorbars:
        # abs colorbar
        norm = colors.Normalize(vmin=np.min(np.abs(array)), vmax=np.max(np.abs(array)))
        cb0 = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cm.gray))
        cb0.set_label("abs", rotation=0, ha="center", va="top")
        cb0.ax.yaxis.set_label_coords(0.5, -0.03)
        abs_sampling = np.linspace(np.min(np.abs(array)),np.max(np.abs(array)),num=num_abs)
        str_abs_sampling = ["{:.2E}".format(n) for n in abs_sampling]
        scaled_vals = scale01(abs_sampling,abs_scaling)
        cb0.set_ticks([0.0, *scaled_vals, 1.0])
        cb0.set_ticklabels(str_abs_sampling)

        # arg colorbar
        # create new colormap
        z = np.exp(1j * np.linspace(-np.pi, np.pi, 256))
        rgb_vals = get_srgb1(z, abs_scaling=abs_scaling, colorspace=colorspace)
        rgba_vals = np.pad(rgb_vals, ((0, 0), (0, 1)), constant_values=1.0)
        newcmp = colors.ListedColormap(rgba_vals)
        #
        norm = colors.Normalize(vmin=-np.pi, vmax=np.pi)
        cb1 = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=newcmp))
        cb1.set_label("arg", rotation=0, ha="center", va="top")
        cb1.ax.yaxis.set_label_coords(0.5, -0.03)
        cb1.set_ticks([-np.pi, -np.pi / 2, 0, +np.pi / 2, np.pi])
        cb1.set_ticklabels(["-π", "-π/2", "0", "π/2", "π"])

from pyxem.utils.pixelated_stem_tools import _get_rgb_phase_magnitude_array
def make_color_wheel(ax, rotation=None):
    x, y = np.mgrid[-2.0:2.0:500j, -2.0:2.0:500j]
    r = (x ** 2 + y ** 2) ** 0.5
    t = np.arctan2(x, y)
    del x, y
    if rotation is not None:
        t += math.radians(rotation)
        t = (t + np.pi) % (2 * np.pi) - np.pi

    r_masked = np.ma.masked_where((2.0 < r) | (r < 1.0), r)
    r_masked -= 1.0

    mask = r_masked.mask
    r_masked.data[r_masked.mask] = r_masked.mean()
    rgb_array = domain_coloring(r_masked.data)
    rgb_array = np.dstack((rgb_array, np.invert(mask)))

    ax.imshow(rgb_array, interpolation="quadric", origin="lower")
    ax.set_axis_off()
fig, ax = plt.subplots(1, 1, figsize=(7, 7))
ax_indicator = fig.add_subplot(331)
make_color_wheel(ax_indicator)
plt.show()
r_masked.data
from abtem.visualize.utils import domain_coloring
def plot_colors(
                array,
                # extent,
                add_colorbars: bool = True,
                num_abs = 5
                ):
    from cplot._colors import get_srgb1,scale01
    from matplotlib import colors,cm
    plt.imshow(
        domain_coloring(array),
        # extent=extent,
        interpolation="nearest",
        origin="lower",
        aspect="equal",
    )

    if add_colorbars:
        # abs colorbar
        norm = colors.Normalize(vmin=np.min(np.abs(array)), vmax=np.max(np.abs(array)))
        cb0 = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cm.gray))
        cb0.set_label("abs", rotation=0, ha="center", va="top")
        cb0.ax.yaxis.set_label_coords(0.5, -0.03)
        abs_sampling = np.linspace(np.min(np.abs(array)),np.max(np.abs(array)),num=num_abs)
        str_abs_sampling = ["{:.2E}".format(n) for n in abs_sampling]
        scaled_vals = scale01(abs_sampling,abs_scaling)
        cb0.set_ticks([0.0, *scaled_vals, 1.0])
        cb0.set_ticklabels(str_abs_sampling)

        # arg colorbar
        # create new colormap
        z = np.exp(1j * np.linspace(-np.pi, np.pi, 256))
        rgb_vals = get_srgb1(z, abs_scaling=abs_scaling, colorspace=colorspace)
        rgba_vals = np.pad(rgb_vals, ((0, 0), (0, 1)), constant_values=1.0)
        newcmp = colors.ListedColormap(rgba_vals)
        #
        norm = colors.Normalize(vmin=-np.pi, vmax=np.pi)
        cb1 = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=newcmp))
        cb1.set_label("arg", rotation=0, ha="center", va="top")
        cb1.ax.yaxis.set_label_coords(0.5, -0.03)
        cb1.set_ticks([-np.pi, -np.pi / 2, 0, +np.pi / 2, np.pi])
        cb1.set_ticklabels(["-π", "-π/2", "0", "π/2", "π"])

from pyxem.utils.pixelated_stem_tools import _get_rgb_phase_magnitude_array
def make_color_wheel(ax, rotation=None):
    x, y = np.mgrid[-2.0:2.0:500j, -2.0:2.0:500j]
    r = (x ** 2 + y ** 2) ** 0.5
    t = np.arctan2(x, y)
    del x, y
    if rotation is not None:
        t += math.radians(rotation)
        t = (t + np.pi) % (2 * np.pi) - np.pi

    r_masked = np.ma.masked_where((2.0 < r) | (r < 1.0), r)
    r_masked -= 1.0

    mask = r_masked.mask
    r_masked.data[r_masked.mask] = r_masked.mean()
    rgb_array = domain_coloring(r_masked.data)
    rgb_array = np.dstack((rgb_array, np.invert(mask)))

    ax.imshow(rgb_array, interpolation="quadric", origin="lower")
    ax.set_axis_off()
fig, ax = plt.subplots(1, 1, figsize=(7, 7))
ax_indicator = fig.add_subplot(331)
make_color_wheel(ax_indicator)
plt.show()
from abtem.visualize.utils import domain_coloring
def plot_colors(
                array,
                # extent,
                add_colorbars: bool = True,
                num_abs = 5
                ):
    from cplot._colors import get_srgb1,scale01
    from matplotlib import colors,cm
    plt.imshow(
        domain_coloring(array),
        # extent=extent,
        interpolation="nearest",
        origin="lower",
        aspect="equal",
    )

    if add_colorbars:
        # abs colorbar
        norm = colors.Normalize(vmin=np.min(np.abs(array)), vmax=np.max(np.abs(array)))
        cb0 = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cm.gray))
        cb0.set_label("abs", rotation=0, ha="center", va="top")
        cb0.ax.yaxis.set_label_coords(0.5, -0.03)
        abs_sampling = np.linspace(np.min(np.abs(array)),np.max(np.abs(array)),num=num_abs)
        str_abs_sampling = ["{:.2E}".format(n) for n in abs_sampling]
        scaled_vals = scale01(abs_sampling,abs_scaling)
        cb0.set_ticks([0.0, *scaled_vals, 1.0])
        cb0.set_ticklabels(str_abs_sampling)

        # arg colorbar
        # create new colormap
        z = np.exp(1j * np.linspace(-np.pi, np.pi, 256))
        rgb_vals = get_srgb1(z, abs_scaling=abs_scaling, colorspace=colorspace)
        rgba_vals = np.pad(rgb_vals, ((0, 0), (0, 1)), constant_values=1.0)
        newcmp = colors.ListedColormap(rgba_vals)
        #
        norm = colors.Normalize(vmin=-np.pi, vmax=np.pi)
        cb1 = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=newcmp))
        cb1.set_label("arg", rotation=0, ha="center", va="top")
        cb1.ax.yaxis.set_label_coords(0.5, -0.03)
        cb1.set_ticks([-np.pi, -np.pi / 2, 0, +np.pi / 2, np.pi])
        cb1.set_ticklabels(["-π", "-π/2", "0", "π/2", "π"])

from pyxem.utils.pixelated_stem_tools import _get_rgb_phase_magnitude_array
def make_color_wheel(ax, rotation=None):
    x, y = np.mgrid[-2.0:2.0:500j, -2.0:2.0:500j]
    r = (x ** 2 + y ** 2) ** 0.5
    t = np.arctan2(x, y)
    del x, y
    if rotation is not None:
        t += math.radians(rotation)
        t = (t + np.pi) % (2 * np.pi) - np.pi

    r_masked = np.ma.masked_where((2.0 < r) | (r < 1.0), r)
    r_masked -= 1.0

    mask = r_masked.mask
    r_masked.data[r_masked.mask] = r_masked.mean()
    rgb_array = domain_coloring(r_masked.data)
    # rgb_array = np.dstack((rgb_array, np.invert(mask)))

    ax.imshow(rgb_array, interpolation="quadric", origin="lower")
    ax.set_axis_off()
fig, ax = plt.subplots(1, 1, figsize=(7, 7))
ax_indicator = fig.add_subplot(331)
make_color_wheel(ax_indicator)
plt.show()
from abtem.visualize.utils import domain_coloring
def plot_colors(
                array,
                # extent,
                add_colorbars: bool = True,
                num_abs = 5
                ):
    from cplot._colors import get_srgb1,scale01
    from matplotlib import colors,cm
    plt.imshow(
        domain_coloring(array),
        # extent=extent,
        interpolation="nearest",
        origin="lower",
        aspect="equal",
    )

    if add_colorbars:
        # abs colorbar
        norm = colors.Normalize(vmin=np.min(np.abs(array)), vmax=np.max(np.abs(array)))
        cb0 = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cm.gray))
        cb0.set_label("abs", rotation=0, ha="center", va="top")
        cb0.ax.yaxis.set_label_coords(0.5, -0.03)
        abs_sampling = np.linspace(np.min(np.abs(array)),np.max(np.abs(array)),num=num_abs)
        str_abs_sampling = ["{:.2E}".format(n) for n in abs_sampling]
        scaled_vals = scale01(abs_sampling,abs_scaling)
        cb0.set_ticks([0.0, *scaled_vals, 1.0])
        cb0.set_ticklabels(str_abs_sampling)

        # arg colorbar
        # create new colormap
        z = np.exp(1j * np.linspace(-np.pi, np.pi, 256))
        rgb_vals = get_srgb1(z, abs_scaling=abs_scaling, colorspace=colorspace)
        rgba_vals = np.pad(rgb_vals, ((0, 0), (0, 1)), constant_values=1.0)
        newcmp = colors.ListedColormap(rgba_vals)
        #
        norm = colors.Normalize(vmin=-np.pi, vmax=np.pi)
        cb1 = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=newcmp))
        cb1.set_label("arg", rotation=0, ha="center", va="top")
        cb1.ax.yaxis.set_label_coords(0.5, -0.03)
        cb1.set_ticks([-np.pi, -np.pi / 2, 0, +np.pi / 2, np.pi])
        cb1.set_ticklabels(["-π", "-π/2", "0", "π/2", "π"])

from pyxem.utils.pixelated_stem_tools import _get_rgb_phase_magnitude_array
def make_color_wheel(ax, rotation=None):
    x, y = np.mgrid[-2.0:2.0:500j, -2.0:2.0:500j]
    r = (x ** 2 + y ** 2) ** 0.5
    t = np.arctan2(x, y)
    del x, y
    if rotation is not None:
        t += math.radians(rotation)
        t = (t + np.pi) % (2 * np.pi) - np.pi

    r_masked = np.ma.masked_where((2.0 < r) | (r < 1.0), r)
    r_masked -= 1.0

    mask = r_masked.mask
    r_masked.data[r_masked.mask] = r_masked.mean()
    rgb_array = domain_coloring(r_masked.data)
    rgb_array = np.dstack((rgb_array, np.invert(mask)))

    ax.imshow(rgb_array, interpolation="quadric", origin="lower")
    ax.set_axis_off()
fig, ax = plt.subplots(1, 1, figsize=(7, 7))
ax_indicator = fig.add_subplot(331)
make_color_wheel(ax_indicator)
plt.show()
from abtem.visualize.utils import domain_coloring
def plot_colors(
                array,
                # extent,
                add_colorbars: bool = True,
                num_abs = 5
                ):
    from cplot._colors import get_srgb1,scale01
    from matplotlib import colors,cm
    plt.imshow(
        domain_coloring(array),
        # extent=extent,
        interpolation="nearest",
        origin="lower",
        aspect="equal",
    )

    if add_colorbars:
        # abs colorbar
        norm = colors.Normalize(vmin=np.min(np.abs(array)), vmax=np.max(np.abs(array)))
        cb0 = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cm.gray))
        cb0.set_label("abs", rotation=0, ha="center", va="top")
        cb0.ax.yaxis.set_label_coords(0.5, -0.03)
        abs_sampling = np.linspace(np.min(np.abs(array)),np.max(np.abs(array)),num=num_abs)
        str_abs_sampling = ["{:.2E}".format(n) for n in abs_sampling]
        scaled_vals = scale01(abs_sampling,abs_scaling)
        cb0.set_ticks([0.0, *scaled_vals, 1.0])
        cb0.set_ticklabels(str_abs_sampling)

        # arg colorbar
        # create new colormap
        z = np.exp(1j * np.linspace(-np.pi, np.pi, 256))
        rgb_vals = get_srgb1(z, abs_scaling=abs_scaling, colorspace=colorspace)
        rgba_vals = np.pad(rgb_vals, ((0, 0), (0, 1)), constant_values=1.0)
        newcmp = colors.ListedColormap(rgba_vals)
        #
        norm = colors.Normalize(vmin=-np.pi, vmax=np.pi)
        cb1 = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=newcmp))
        cb1.set_label("arg", rotation=0, ha="center", va="top")
        cb1.ax.yaxis.set_label_coords(0.5, -0.03)
        cb1.set_ticks([-np.pi, -np.pi / 2, 0, +np.pi / 2, np.pi])
        cb1.set_ticklabels(["-π", "-π/2", "0", "π/2", "π"])

from pyxem.utils.pixelated_stem_tools import _get_rgb_phase_magnitude_array
def make_color_wheel(ax, rotation=None):
    x, y = np.mgrid[-2.0:2.0:500j, -2.0:2.0:500j]
    r = (x ** 2 + y ** 2) ** 0.5
    t = np.arctan2(x, y)
    del x, y
    if rotation is not None:
        t += math.radians(rotation)
        t = (t + np.pi) % (2 * np.pi) - np.pi

    r_masked = np.ma.masked_where((2.0 < r) | (r < 1.0), r)
    r_masked -= 1.0

    mask = r_masked.mask
    r_masked.data[r_masked.mask] = r_masked.mean()
    rgb_array = domain_coloring(np.dot(r_masked.data,np.cos(t))+1j*np.dot(r_masked.data,np.sin(t)))
    rgb_array = np.dstack((rgb_array, np.invert(mask)))

    ax.imshow(rgb_array, interpolation="quadric", origin="lower")
    ax.set_axis_off()
fig, ax = plt.subplots(1, 1, figsize=(7, 7))
ax_indicator = fig.add_subplot(331)
make_color_wheel(ax_indicator)
plt.show()
from abtem.visualize.utils import domain_coloring
def plot_colors(
                array,
                # extent,
                add_colorbars: bool = True,
                num_abs = 5
                ):
    from cplot._colors import get_srgb1,scale01
    from matplotlib import colors,cm
    plt.imshow(
        domain_coloring(array),
        # extent=extent,
        interpolation="nearest",
        origin="lower",
        aspect="equal",
    )

    if add_colorbars:
        # abs colorbar
        norm = colors.Normalize(vmin=np.min(np.abs(array)), vmax=np.max(np.abs(array)))
        cb0 = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cm.gray))
        cb0.set_label("abs", rotation=0, ha="center", va="top")
        cb0.ax.yaxis.set_label_coords(0.5, -0.03)
        abs_sampling = np.linspace(np.min(np.abs(array)),np.max(np.abs(array)),num=num_abs)
        str_abs_sampling = ["{:.2E}".format(n) for n in abs_sampling]
        scaled_vals = scale01(abs_sampling,abs_scaling)
        cb0.set_ticks([0.0, *scaled_vals, 1.0])
        cb0.set_ticklabels(str_abs_sampling)

        # arg colorbar
        # create new colormap
        z = np.exp(1j * np.linspace(-np.pi, np.pi, 256))
        rgb_vals = get_srgb1(z, abs_scaling=abs_scaling, colorspace=colorspace)
        rgba_vals = np.pad(rgb_vals, ((0, 0), (0, 1)), constant_values=1.0)
        newcmp = colors.ListedColormap(rgba_vals)
        #
        norm = colors.Normalize(vmin=-np.pi, vmax=np.pi)
        cb1 = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=newcmp))
        cb1.set_label("arg", rotation=0, ha="center", va="top")
        cb1.ax.yaxis.set_label_coords(0.5, -0.03)
        cb1.set_ticks([-np.pi, -np.pi / 2, 0, +np.pi / 2, np.pi])
        cb1.set_ticklabels(["-π", "-π/2", "0", "π/2", "π"])

from pyxem.utils.pixelated_stem_tools import _get_rgb_phase_magnitude_array
def make_color_wheel(ax, rotation=None):
    x, y = np.mgrid[-2.0:2.0:500j, -2.0:2.0:500j]
    r = (x ** 2 + y ** 2) ** 0.5
    t = np.arctan2(x, y)
    del x, y
    if rotation is not None:
        t += math.radians(rotation)
        t = (t + np.pi) % (2 * np.pi) - np.pi

    r_masked = np.ma.masked_where((2.0 < r) | (r < 1.0), r)
    r_masked -= 1.0

    mask = r_masked.mask
    r_masked.data[r_masked.mask] = r_masked.mean()
    rgb_array = domain_coloring(np.dot(np.cos(t))+1j*np.dot(np.sin(t)))
    rgb_array = np.dstack((rgb_array, np.invert(mask)))

    ax.imshow(rgb_array, interpolation="quadric", origin="lower")
    ax.set_axis_off()
fig, ax = plt.subplots(1, 1, figsize=(7, 7))
ax_indicator = fig.add_subplot(331)
make_color_wheel(ax_indicator)
plt.show()
from abtem.visualize.utils import domain_coloring
def plot_colors(
                array,
                # extent,
                add_colorbars: bool = True,
                num_abs = 5
                ):
    from cplot._colors import get_srgb1,scale01
    from matplotlib import colors,cm
    plt.imshow(
        domain_coloring(array),
        # extent=extent,
        interpolation="nearest",
        origin="lower",
        aspect="equal",
    )

    if add_colorbars:
        # abs colorbar
        norm = colors.Normalize(vmin=np.min(np.abs(array)), vmax=np.max(np.abs(array)))
        cb0 = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cm.gray))
        cb0.set_label("abs", rotation=0, ha="center", va="top")
        cb0.ax.yaxis.set_label_coords(0.5, -0.03)
        abs_sampling = np.linspace(np.min(np.abs(array)),np.max(np.abs(array)),num=num_abs)
        str_abs_sampling = ["{:.2E}".format(n) for n in abs_sampling]
        scaled_vals = scale01(abs_sampling,abs_scaling)
        cb0.set_ticks([0.0, *scaled_vals, 1.0])
        cb0.set_ticklabels(str_abs_sampling)

        # arg colorbar
        # create new colormap
        z = np.exp(1j * np.linspace(-np.pi, np.pi, 256))
        rgb_vals = get_srgb1(z, abs_scaling=abs_scaling, colorspace=colorspace)
        rgba_vals = np.pad(rgb_vals, ((0, 0), (0, 1)), constant_values=1.0)
        newcmp = colors.ListedColormap(rgba_vals)
        #
        norm = colors.Normalize(vmin=-np.pi, vmax=np.pi)
        cb1 = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=newcmp))
        cb1.set_label("arg", rotation=0, ha="center", va="top")
        cb1.ax.yaxis.set_label_coords(0.5, -0.03)
        cb1.set_ticks([-np.pi, -np.pi / 2, 0, +np.pi / 2, np.pi])
        cb1.set_ticklabels(["-π", "-π/2", "0", "π/2", "π"])

from pyxem.utils.pixelated_stem_tools import _get_rgb_phase_magnitude_array
def make_color_wheel(ax, rotation=None):
    x, y = np.mgrid[-2.0:2.0:500j, -2.0:2.0:500j]
    r = (x ** 2 + y ** 2) ** 0.5
    t = np.arctan2(x, y)
    del x, y
    if rotation is not None:
        t += math.radians(rotation)
        t = (t + np.pi) % (2 * np.pi) - np.pi

    r_masked = np.ma.masked_where((2.0 < r) | (r < 1.0), r)
    r_masked -= 1.0

    mask = r_masked.mask
    r_masked.data[r_masked.mask] = r_masked.mean()
    rgb_array = domain_coloring(np.dot(1,np.cos(t))+1j*np.dot(1,np.sin(t)))
    rgb_array = np.dstack((rgb_array, np.invert(mask)))

    ax.imshow(rgb_array, interpolation="quadric", origin="lower")
    ax.set_axis_off()
fig, ax = plt.subplots(1, 1, figsize=(7, 7))
ax_indicator = fig.add_subplot(331)
make_color_wheel(ax_indicator)
plt.show()
from abtem.visualize.utils import domain_coloring
def plot_colors(
                array,
                # extent,
                add_colorbars: bool = True,
                num_abs = 5
                ):
    from cplot._colors import get_srgb1,scale01
    from matplotlib import colors,cm
    plt.imshow(
        domain_coloring(array),
        # extent=extent,
        interpolation="nearest",
        origin="lower",
        aspect="equal",
    )

    if add_colorbars:
        # abs colorbar
        norm = colors.Normalize(vmin=np.min(np.abs(array)), vmax=np.max(np.abs(array)))
        cb0 = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cm.gray))
        cb0.set_label("abs", rotation=0, ha="center", va="top")
        cb0.ax.yaxis.set_label_coords(0.5, -0.03)
        abs_sampling = np.linspace(np.min(np.abs(array)),np.max(np.abs(array)),num=num_abs)
        str_abs_sampling = ["{:.2E}".format(n) for n in abs_sampling]
        scaled_vals = scale01(abs_sampling,abs_scaling)
        cb0.set_ticks([0.0, *scaled_vals, 1.0])
        cb0.set_ticklabels(str_abs_sampling)

        # arg colorbar
        # create new colormap
        z = np.exp(1j * np.linspace(-np.pi, np.pi, 256))
        rgb_vals = get_srgb1(z, abs_scaling=abs_scaling, colorspace=colorspace)
        rgba_vals = np.pad(rgb_vals, ((0, 0), (0, 1)), constant_values=1.0)
        newcmp = colors.ListedColormap(rgba_vals)
        #
        norm = colors.Normalize(vmin=-np.pi, vmax=np.pi)
        cb1 = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=newcmp))
        cb1.set_label("arg", rotation=0, ha="center", va="top")
        cb1.ax.yaxis.set_label_coords(0.5, -0.03)
        cb1.set_ticks([-np.pi, -np.pi / 2, 0, +np.pi / 2, np.pi])
        cb1.set_ticklabels(["-π", "-π/2", "0", "π/2", "π"])

from pyxem.utils.pixelated_stem_tools import _get_rgb_phase_magnitude_array
def make_color_wheel(ax, rotation=None):
    x, y = np.mgrid[-2.0:2.0:500j, -2.0:2.0:500j]
    r = (x ** 2 + y ** 2) ** 0.5
    t = np.arctan2(x, y)
    del x, y
    if rotation is not None:
        t += math.radians(rotation)
        t = (t + np.pi) % (2 * np.pi) - np.pi

    r_masked = np.ma.masked_where((2.0 < r) | (r < 1.0), r)
    r_masked -= 1.0

    mask = r_masked.mask
    r_masked.data[r_masked.mask] = r_masked.mean()
    rgb_array = domain_coloring(np.dot(r_masked,np.cos(t))+1j*np.dot(r_masked,np.sin(t)))
    # rgb_array = np.dstack((rgb_array, np.invert(mask)))

    ax.imshow(rgb_array, interpolation="quadric", origin="lower")
    ax.set_axis_off()
fig, ax = plt.subplots(1, 1, figsize=(7, 7))
ax_indicator = fig.add_subplot(331)
make_color_wheel(ax_indicator)
plt.show()
from abtem.visualize.utils import domain_coloring
def plot_colors(
                array,
                # extent,
                add_colorbars: bool = True,
                num_abs = 5
                ):
    from cplot._colors import get_srgb1,scale01
    from matplotlib import colors,cm
    plt.imshow(
        domain_coloring(array),
        # extent=extent,
        interpolation="nearest",
        origin="lower",
        aspect="equal",
    )

    if add_colorbars:
        # abs colorbar
        norm = colors.Normalize(vmin=np.min(np.abs(array)), vmax=np.max(np.abs(array)))
        cb0 = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cm.gray))
        cb0.set_label("abs", rotation=0, ha="center", va="top")
        cb0.ax.yaxis.set_label_coords(0.5, -0.03)
        abs_sampling = np.linspace(np.min(np.abs(array)),np.max(np.abs(array)),num=num_abs)
        str_abs_sampling = ["{:.2E}".format(n) for n in abs_sampling]
        scaled_vals = scale01(abs_sampling,abs_scaling)
        cb0.set_ticks([0.0, *scaled_vals, 1.0])
        cb0.set_ticklabels(str_abs_sampling)

        # arg colorbar
        # create new colormap
        z = np.exp(1j * np.linspace(-np.pi, np.pi, 256))
        rgb_vals = get_srgb1(z, abs_scaling=abs_scaling, colorspace=colorspace)
        rgba_vals = np.pad(rgb_vals, ((0, 0), (0, 1)), constant_values=1.0)
        newcmp = colors.ListedColormap(rgba_vals)
        #
        norm = colors.Normalize(vmin=-np.pi, vmax=np.pi)
        cb1 = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=newcmp))
        cb1.set_label("arg", rotation=0, ha="center", va="top")
        cb1.ax.yaxis.set_label_coords(0.5, -0.03)
        cb1.set_ticks([-np.pi, -np.pi / 2, 0, +np.pi / 2, np.pi])
        cb1.set_ticklabels(["-π", "-π/2", "0", "π/2", "π"])

from pyxem.utils.pixelated_stem_tools import _get_rgb_phase_magnitude_array
def make_color_wheel(ax, rotation=None):
    x, y = np.mgrid[-2.0:2.0:500j, -2.0:2.0:500j]
    r = (x ** 2 + y ** 2) ** 0.5
    t = np.arctan2(x, y)
    del x, y
    if rotation is not None:
        t += math.radians(rotation)
        t = (t + np.pi) % (2 * np.pi) - np.pi

    r_masked = np.ma.masked_where((2.0 < r) | (r < 1.0), r)
    r_masked -= 1.0

    mask = r_masked.mask
    r_masked.data[r_masked.mask] = r_masked.mean()
    rgb_array = domain_coloring(np.dot(r_masked,np.cos(t))+1j*np.dot(r_masked,np.sin(t)))
    rgb_array = np.dstack((rgb_array, np.invert(mask)))

    ax.imshow(rgb_array, interpolation="quadric", origin="lower")
    ax.set_axis_off()
fig, ax = plt.subplots(1, 1, figsize=(7, 7))
ax_indicator = fig.add_subplot(331)
make_color_wheel(ax_indicator)
plt.show()
from abtem.visualize.utils import domain_coloring
def plot_colors(
                array,
                # extent,
                add_colorbars: bool = True,
                num_abs = 5
                ):
    from cplot._colors import get_srgb1,scale01
    from matplotlib import colors,cm
    plt.imshow(
        domain_coloring(array),
        # extent=extent,
        interpolation="nearest",
        origin="lower",
        aspect="equal",
    )

    if add_colorbars:
        # abs colorbar
        norm = colors.Normalize(vmin=np.min(np.abs(array)), vmax=np.max(np.abs(array)))
        cb0 = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cm.gray))
        cb0.set_label("abs", rotation=0, ha="center", va="top")
        cb0.ax.yaxis.set_label_coords(0.5, -0.03)
        abs_sampling = np.linspace(np.min(np.abs(array)),np.max(np.abs(array)),num=num_abs)
        str_abs_sampling = ["{:.2E}".format(n) for n in abs_sampling]
        scaled_vals = scale01(abs_sampling,abs_scaling)
        cb0.set_ticks([0.0, *scaled_vals, 1.0])
        cb0.set_ticklabels(str_abs_sampling)

        # arg colorbar
        # create new colormap
        z = np.exp(1j * np.linspace(-np.pi, np.pi, 256))
        rgb_vals = get_srgb1(z, abs_scaling=abs_scaling, colorspace=colorspace)
        rgba_vals = np.pad(rgb_vals, ((0, 0), (0, 1)), constant_values=1.0)
        newcmp = colors.ListedColormap(rgba_vals)
        #
        norm = colors.Normalize(vmin=-np.pi, vmax=np.pi)
        cb1 = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=newcmp))
        cb1.set_label("arg", rotation=0, ha="center", va="top")
        cb1.ax.yaxis.set_label_coords(0.5, -0.03)
        cb1.set_ticks([-np.pi, -np.pi / 2, 0, +np.pi / 2, np.pi])
        cb1.set_ticklabels(["-π", "-π/2", "0", "π/2", "π"])

from pyxem.utils.pixelated_stem_tools import _get_rgb_phase_magnitude_array
def make_color_wheel(ax, rotation=None):
    x, y = np.mgrid[-2.0:2.0:500j, -2.0:2.0:500j]
    r = (x ** 2 + y ** 2) ** 0.5
    t = np.arctan2(x, y)
    del x, y
    if rotation is not None:
        t += math.radians(rotation)
        t = (t + np.pi) % (2 * np.pi) - np.pi

    r_masked = np.ma.masked_where((2.0 < r) | (r < 1.0), r)
    r_masked -= 1.0

    mask = r_masked.mask
    r_masked.data[r_masked.mask] = r_masked.mean()
    rgb_array = domain_coloring(np.dot(r_masked,np.cos(t))+1j*np.dot(r_masked,np.sin(t)))
    rgb_array = np.dstack((rgb_array, np.invert(mask)))

    ax.imshow(rgb_array, interpolation="quadric", origin="lower")
    ax.set_axis_off()
import debugpy
debugpy.debug_this_thread()
fig, ax = plt.subplots(1, 1, figsize=(7, 7))
ax_indicator = fig.add_subplot(331)
make_color_wheel(ax_indicator)
plt.show()
from abtem.visualize.utils import domain_coloring
def plot_colors(
                array,
                # extent,
                add_colorbars: bool = True,
                num_abs = 5
                ):
    from cplot._colors import get_srgb1,scale01
    from matplotlib import colors,cm
    plt.imshow(
        domain_coloring(array),
        # extent=extent,
        interpolation="nearest",
        origin="lower",
        aspect="equal",
    )

    if add_colorbars:
        # abs colorbar
        norm = colors.Normalize(vmin=np.min(np.abs(array)), vmax=np.max(np.abs(array)))
        cb0 = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cm.gray))
        cb0.set_label("abs", rotation=0, ha="center", va="top")
        cb0.ax.yaxis.set_label_coords(0.5, -0.03)
        abs_sampling = np.linspace(np.min(np.abs(array)),np.max(np.abs(array)),num=num_abs)
        str_abs_sampling = ["{:.2E}".format(n) for n in abs_sampling]
        scaled_vals = scale01(abs_sampling,abs_scaling)
        cb0.set_ticks([0.0, *scaled_vals, 1.0])
        cb0.set_ticklabels(str_abs_sampling)

        # arg colorbar
        # create new colormap
        z = np.exp(1j * np.linspace(-np.pi, np.pi, 256))
        rgb_vals = get_srgb1(z, abs_scaling=abs_scaling, colorspace=colorspace)
        rgba_vals = np.pad(rgb_vals, ((0, 0), (0, 1)), constant_values=1.0)
        newcmp = colors.ListedColormap(rgba_vals)
        #
        norm = colors.Normalize(vmin=-np.pi, vmax=np.pi)
        cb1 = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=newcmp))
        cb1.set_label("arg", rotation=0, ha="center", va="top")
        cb1.ax.yaxis.set_label_coords(0.5, -0.03)
        cb1.set_ticks([-np.pi, -np.pi / 2, 0, +np.pi / 2, np.pi])
        cb1.set_ticklabels(["-π", "-π/2", "0", "π/2", "π"])

from pyxem.utils.pixelated_stem_tools import _get_rgb_phase_magnitude_array
def make_color_wheel(ax, rotation=None):
    x, y = np.mgrid[-2.0:2.0:500j, -2.0:2.0:500j]
    r = (x ** 2 + y ** 2) ** 0.5
    t = np.arctan2(x, y)
    del x, y
    if rotation is not None:
        t += math.radians(rotation)
        t = (t + np.pi) % (2 * np.pi) - np.pi

    r_masked = np.ma.masked_where((2.0 < r) | (r < 1.0), r)
    r_masked -= 1.0

    mask = r_masked.mask
    r_masked.data[r_masked.mask] = r_masked.mean()
    rgb_array = domain_coloring(np.dot(r_masked.data,np.exp(1j*t)))
    rgb_array = np.dstack((rgb_array, np.invert(mask)))

    ax.imshow(rgb_array, interpolation="quadric", origin="lower")
    ax.set_axis_off()
from abtem.visualize.utils import domain_coloring
def plot_colors(
                array,
                # extent,
                add_colorbars: bool = True,
                num_abs = 5
                ):
    from cplot._colors import get_srgb1,scale01
    from matplotlib import colors,cm
    plt.imshow(
        domain_coloring(array),
        # extent=extent,
        interpolation="nearest",
        origin="lower",
        aspect="equal",
    )

    if add_colorbars:
        # abs colorbar
        norm = colors.Normalize(vmin=np.min(np.abs(array)), vmax=np.max(np.abs(array)))
        cb0 = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cm.gray))
        cb0.set_label("abs", rotation=0, ha="center", va="top")
        cb0.ax.yaxis.set_label_coords(0.5, -0.03)
        abs_sampling = np.linspace(np.min(np.abs(array)),np.max(np.abs(array)),num=num_abs)
        str_abs_sampling = ["{:.2E}".format(n) for n in abs_sampling]
        scaled_vals = scale01(abs_sampling,abs_scaling)
        cb0.set_ticks([0.0, *scaled_vals, 1.0])
        cb0.set_ticklabels(str_abs_sampling)

        # arg colorbar
        # create new colormap
        z = np.exp(1j * np.linspace(-np.pi, np.pi, 256))
        rgb_vals = get_srgb1(z, abs_scaling=abs_scaling, colorspace=colorspace)
        rgba_vals = np.pad(rgb_vals, ((0, 0), (0, 1)), constant_values=1.0)
        newcmp = colors.ListedColormap(rgba_vals)
        #
        norm = colors.Normalize(vmin=-np.pi, vmax=np.pi)
        cb1 = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=newcmp))
        cb1.set_label("arg", rotation=0, ha="center", va="top")
        cb1.ax.yaxis.set_label_coords(0.5, -0.03)
        cb1.set_ticks([-np.pi, -np.pi / 2, 0, +np.pi / 2, np.pi])
        cb1.set_ticklabels(["-π", "-π/2", "0", "π/2", "π"])

from pyxem.utils.pixelated_stem_tools import _get_rgb_phase_magnitude_array
def make_color_wheel(ax, rotation=None):
    x, y = np.mgrid[-2.0:2.0:500j, -2.0:2.0:500j]
    r = (x ** 2 + y ** 2) ** 0.5
    t = np.arctan2(x, y)
    del x, y
    if rotation is not None:
        t += math.radians(rotation)
        t = (t + np.pi) % (2 * np.pi) - np.pi

    r_masked = np.ma.masked_where((2.0 < r) | (r < 1.0), r)
    r_masked -= 1.0

    mask = r_masked.mask
    r_masked.data[r_masked.mask] = r_masked.mean()
    rgb_array = domain_coloring(np.dot(r_masked.data,np.exp(1j*t)))
    rgb_array = np.dstack((rgb_array, np.invert(mask)))

    ax.imshow(rgb_array, interpolation="quadric", origin="lower")
    ax.set_axis_off()
fig, ax = plt.subplots(1, 1, figsize=(7, 7))
ax_indicator = fig.add_subplot(331)
make_color_wheel(ax_indicator)
plt.show()
from abtem.visualize.utils import domain_coloring
def plot_colors(
                array,
                # extent,
                add_colorbars: bool = True,
                num_abs = 5
                ):
    from cplot._colors import get_srgb1,scale01
    from matplotlib import colors,cm
    plt.imshow(
        domain_coloring(array),
        # extent=extent,
        interpolation="nearest",
        origin="lower",
        aspect="equal",
    )

    if add_colorbars:
        # abs colorbar
        norm = colors.Normalize(vmin=np.min(np.abs(array)), vmax=np.max(np.abs(array)))
        cb0 = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cm.gray))
        cb0.set_label("abs", rotation=0, ha="center", va="top")
        cb0.ax.yaxis.set_label_coords(0.5, -0.03)
        abs_sampling = np.linspace(np.min(np.abs(array)),np.max(np.abs(array)),num=num_abs)
        str_abs_sampling = ["{:.2E}".format(n) for n in abs_sampling]
        scaled_vals = scale01(abs_sampling,abs_scaling)
        cb0.set_ticks([0.0, *scaled_vals, 1.0])
        cb0.set_ticklabels(str_abs_sampling)

        # arg colorbar
        # create new colormap
        z = np.exp(1j * np.linspace(-np.pi, np.pi, 256))
        rgb_vals = get_srgb1(z, abs_scaling=abs_scaling, colorspace=colorspace)
        rgba_vals = np.pad(rgb_vals, ((0, 0), (0, 1)), constant_values=1.0)
        newcmp = colors.ListedColormap(rgba_vals)
        #
        norm = colors.Normalize(vmin=-np.pi, vmax=np.pi)
        cb1 = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=newcmp))
        cb1.set_label("arg", rotation=0, ha="center", va="top")
        cb1.ax.yaxis.set_label_coords(0.5, -0.03)
        cb1.set_ticks([-np.pi, -np.pi / 2, 0, +np.pi / 2, np.pi])
        cb1.set_ticklabels(["-π", "-π/2", "0", "π/2", "π"])

from pyxem.utils.pixelated_stem_tools import _get_rgb_phase_magnitude_array
def make_color_wheel(ax, rotation=None):
    x, y = np.mgrid[-2.0:2.0:500, -2.0:2.0:500j]
    r = (x ** 2 + y ** 2) ** 0.5
    t = np.arctan2(x, y)
    del x, y
    if rotation is not None:
        t += math.radians(rotation)
        t = (t + np.pi) % (2 * np.pi) - np.pi

    r_masked = np.ma.masked_where((2.0 < r) | (r < 1.0), r)
    r_masked -= 1.0

    mask = r_masked.mask
    r_masked.data[r_masked.mask] = r_masked.mean()
    rgb_array = domain_coloring(np.dot(r_masked.data,np.exp(1j*t)))
    rgb_array = np.dstack((rgb_array, np.invert(mask)))

    ax.imshow(rgb_array, interpolation="quadric", origin="lower")
    ax.set_axis_off()
fig, ax = plt.subplots(1, 1, figsize=(7, 7))
ax_indicator = fig.add_subplot(331)
make_color_wheel(ax_indicator)
plt.show()
from abtem.visualize.utils import domain_coloring
def plot_colors(
                array,
                # extent,
                add_colorbars: bool = True,
                num_abs = 5
                ):
    from cplot._colors import get_srgb1,scale01
    from matplotlib import colors,cm
    plt.imshow(
        domain_coloring(array),
        # extent=extent,
        interpolation="nearest",
        origin="lower",
        aspect="equal",
    )

    if add_colorbars:
        # abs colorbar
        norm = colors.Normalize(vmin=np.min(np.abs(array)), vmax=np.max(np.abs(array)))
        cb0 = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cm.gray))
        cb0.set_label("abs", rotation=0, ha="center", va="top")
        cb0.ax.yaxis.set_label_coords(0.5, -0.03)
        abs_sampling = np.linspace(np.min(np.abs(array)),np.max(np.abs(array)),num=num_abs)
        str_abs_sampling = ["{:.2E}".format(n) for n in abs_sampling]
        scaled_vals = scale01(abs_sampling,abs_scaling)
        cb0.set_ticks([0.0, *scaled_vals, 1.0])
        cb0.set_ticklabels(str_abs_sampling)

        # arg colorbar
        # create new colormap
        z = np.exp(1j * np.linspace(-np.pi, np.pi, 256))
        rgb_vals = get_srgb1(z, abs_scaling=abs_scaling, colorspace=colorspace)
        rgba_vals = np.pad(rgb_vals, ((0, 0), (0, 1)), constant_values=1.0)
        newcmp = colors.ListedColormap(rgba_vals)
        #
        norm = colors.Normalize(vmin=-np.pi, vmax=np.pi)
        cb1 = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=newcmp))
        cb1.set_label("arg", rotation=0, ha="center", va="top")
        cb1.ax.yaxis.set_label_coords(0.5, -0.03)
        cb1.set_ticks([-np.pi, -np.pi / 2, 0, +np.pi / 2, np.pi])
        cb1.set_ticklabels(["-π", "-π/2", "0", "π/2", "π"])

from pyxem.utils.pixelated_stem_tools import _get_rgb_phase_magnitude_array
def make_color_wheel(ax, rotation=None):
    x, y = np.mgrid[-2.0:2.0:500j, -2.0:2.0:500j]
    r = (x ** 2 + y ** 2) ** 0.5
    t = np.arctan2(x, y)
    del x, y
    if rotation is not None:
        t += math.radians(rotation)
        t = (t + np.pi) % (2 * np.pi) - np.pi

    r_masked = np.ma.masked_where((2.0 < r) | (r < 1.0), r)
    r_masked -= 1.0

    mask = r_masked.mask
    r_masked.data[r_masked.mask] = r_masked.mean()
    rgb_array = domain_coloring(np.dot(r_masked.data,np.exp(1j*t)))
    rgb_array = np.dstack((rgb_array, np.invert(mask)))

    ax.imshow(rgb_array, interpolation="quadric", origin="lower")
    ax.set_axis_off()
import debugpy
debugpy.debug_this_thread()
fig, ax = plt.subplots(1, 1, figsize=(7, 7))
ax_indicator = fig.add_subplot(331)
make_color_wheel(ax_indicator)
plt.show()
from abtem.visualize.utils import domain_coloring
def plot_colors(
                array,
                # extent,
                add_colorbars: bool = True,
                num_abs = 5
                ):
    from cplot._colors import get_srgb1,scale01
    from matplotlib import colors,cm
    plt.imshow(
        domain_coloring(array),
        # extent=extent,
        interpolation="nearest",
        origin="lower",
        aspect="equal",
    )

    if add_colorbars:
        # abs colorbar
        norm = colors.Normalize(vmin=np.min(np.abs(array)), vmax=np.max(np.abs(array)))
        cb0 = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cm.gray))
        cb0.set_label("abs", rotation=0, ha="center", va="top")
        cb0.ax.yaxis.set_label_coords(0.5, -0.03)
        abs_sampling = np.linspace(np.min(np.abs(array)),np.max(np.abs(array)),num=num_abs)
        str_abs_sampling = ["{:.2E}".format(n) for n in abs_sampling]
        scaled_vals = scale01(abs_sampling,abs_scaling)
        cb0.set_ticks([0.0, *scaled_vals, 1.0])
        cb0.set_ticklabels(str_abs_sampling)

        # arg colorbar
        # create new colormap
        z = np.exp(1j * np.linspace(-np.pi, np.pi, 256))
        rgb_vals = get_srgb1(z, abs_scaling=abs_scaling, colorspace=colorspace)
        rgba_vals = np.pad(rgb_vals, ((0, 0), (0, 1)), constant_values=1.0)
        newcmp = colors.ListedColormap(rgba_vals)
        #
        norm = colors.Normalize(vmin=-np.pi, vmax=np.pi)
        cb1 = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=newcmp))
        cb1.set_label("arg", rotation=0, ha="center", va="top")
        cb1.ax.yaxis.set_label_coords(0.5, -0.03)
        cb1.set_ticks([-np.pi, -np.pi / 2, 0, +np.pi / 2, np.pi])
        cb1.set_ticklabels(["-π", "-π/2", "0", "π/2", "π"])

from pyxem.utils.pixelated_stem_tools import _get_rgb_phase_magnitude_array
def make_color_wheel(ax, rotation=None):
    x, y = np.mgrid[-500:1:500, -500:1:500]
    r = (x ** 2 + y ** 2) ** 0.5
    t = np.arctan2(x, y)
    del x, y
    if rotation is not None:
        t += math.radians(rotation)
        t = (t + np.pi) % (2 * np.pi) - np.pi

    r_masked = np.ma.masked_where((2.0 < r) | (r < 1.0), r)
    r_masked -= 1.0

    mask = r_masked.mask
    r_masked.data[r_masked.mask] = r_masked.mean()
    rgb_array = domain_coloring(np.dot(r_masked.data,np.exp(1j*t)))
    rgb_array = np.dstack((rgb_array, np.invert(mask)))

    ax.imshow(rgb_array, interpolation="quadric", origin="lower")
    ax.set_axis_off()
fig, ax = plt.subplots(1, 1, figsize=(7, 7))
ax_indicator = fig.add_subplot(331)
make_color_wheel(ax_indicator)
plt.show()
from abtem.visualize.utils import domain_coloring
def plot_colors(
                array,
                # extent,
                add_colorbars: bool = True,
                num_abs = 5
                ):
    from cplot._colors import get_srgb1,scale01
    from matplotlib import colors,cm
    plt.imshow(
        domain_coloring(array),
        # extent=extent,
        interpolation="nearest",
        origin="lower",
        aspect="equal",
    )

    if add_colorbars:
        # abs colorbar
        norm = colors.Normalize(vmin=np.min(np.abs(array)), vmax=np.max(np.abs(array)))
        cb0 = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cm.gray))
        cb0.set_label("abs", rotation=0, ha="center", va="top")
        cb0.ax.yaxis.set_label_coords(0.5, -0.03)
        abs_sampling = np.linspace(np.min(np.abs(array)),np.max(np.abs(array)),num=num_abs)
        str_abs_sampling = ["{:.2E}".format(n) for n in abs_sampling]
        scaled_vals = scale01(abs_sampling,abs_scaling)
        cb0.set_ticks([0.0, *scaled_vals, 1.0])
        cb0.set_ticklabels(str_abs_sampling)

        # arg colorbar
        # create new colormap
        z = np.exp(1j * np.linspace(-np.pi, np.pi, 256))
        rgb_vals = get_srgb1(z, abs_scaling=abs_scaling, colorspace=colorspace)
        rgba_vals = np.pad(rgb_vals, ((0, 0), (0, 1)), constant_values=1.0)
        newcmp = colors.ListedColormap(rgba_vals)
        #
        norm = colors.Normalize(vmin=-np.pi, vmax=np.pi)
        cb1 = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=newcmp))
        cb1.set_label("arg", rotation=0, ha="center", va="top")
        cb1.ax.yaxis.set_label_coords(0.5, -0.03)
        cb1.set_ticks([-np.pi, -np.pi / 2, 0, +np.pi / 2, np.pi])
        cb1.set_ticklabels(["-π", "-π/2", "0", "π/2", "π"])

from pyxem.utils.pixelated_stem_tools import _get_rgb_phase_magnitude_array
def make_color_wheel(ax, rotation=None):
    x, y = np.mgrid[-2:0.01:2, -2:0.01:2]
    r = (x ** 2 + y ** 2) ** 0.5
    t = np.arctan2(x, y)
    del x, y
    if rotation is not None:
        t += math.radians(rotation)
        t = (t + np.pi) % (2 * np.pi) - np.pi

    r_masked = np.ma.masked_where((2.0 < r) | (r < 1.0), r)
    r_masked -= 1.0

    mask = r_masked.mask
    r_masked.data[r_masked.mask] = r_masked.mean()
    rgb_array = domain_coloring(np.dot(r_masked.data,np.exp(1j*t)))
    rgb_array = np.dstack((rgb_array, np.invert(mask)))

    ax.imshow(rgb_array, interpolation="quadric", origin="lower")
    ax.set_axis_off()
fig, ax = plt.subplots(1, 1, figsize=(7, 7))
ax_indicator = fig.add_subplot(331)
make_color_wheel(ax_indicator)
plt.show()
from abtem.visualize.utils import domain_coloring
def plot_colors(
                array,
                # extent,
                add_colorbars: bool = True,
                num_abs = 5
                ):
    from cplot._colors import get_srgb1,scale01
    from matplotlib import colors,cm
    plt.imshow(
        domain_coloring(array),
        # extent=extent,
        interpolation="nearest",
        origin="lower",
        aspect="equal",
    )

    if add_colorbars:
        # abs colorbar
        norm = colors.Normalize(vmin=np.min(np.abs(array)), vmax=np.max(np.abs(array)))
        cb0 = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cm.gray))
        cb0.set_label("abs", rotation=0, ha="center", va="top")
        cb0.ax.yaxis.set_label_coords(0.5, -0.03)
        abs_sampling = np.linspace(np.min(np.abs(array)),np.max(np.abs(array)),num=num_abs)
        str_abs_sampling = ["{:.2E}".format(n) for n in abs_sampling]
        scaled_vals = scale01(abs_sampling,abs_scaling)
        cb0.set_ticks([0.0, *scaled_vals, 1.0])
        cb0.set_ticklabels(str_abs_sampling)

        # arg colorbar
        # create new colormap
        z = np.exp(1j * np.linspace(-np.pi, np.pi, 256))
        rgb_vals = get_srgb1(z, abs_scaling=abs_scaling, colorspace=colorspace)
        rgba_vals = np.pad(rgb_vals, ((0, 0), (0, 1)), constant_values=1.0)
        newcmp = colors.ListedColormap(rgba_vals)
        #
        norm = colors.Normalize(vmin=-np.pi, vmax=np.pi)
        cb1 = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=newcmp))
        cb1.set_label("arg", rotation=0, ha="center", va="top")
        cb1.ax.yaxis.set_label_coords(0.5, -0.03)
        cb1.set_ticks([-np.pi, -np.pi / 2, 0, +np.pi / 2, np.pi])
        cb1.set_ticklabels(["-π", "-π/2", "0", "π/2", "π"])

from pyxem.utils.pixelated_stem_tools import _get_rgb_phase_magnitude_array
def make_color_wheel(ax, rotation=None):
    x, y = np.mgrid[-2:0.01:2, -2:0.01:2]
    r = (x ** 2 + y ** 2) ** 0.5
    t = np.arctan2(x, y)
    del x, y
    if rotation is not None:
        t += math.radians(rotation)
        t = (t + np.pi) % (2 * np.pi) - np.pi

    r_masked = np.ma.masked_where((2.0 < r) | (r < 1.0), r)
    r_masked -= 1.0

    mask = r_masked.mask
    r_masked.data[r_masked.mask] = r_masked.mean()
    rgb_array = domain_coloring(np.dot(r_masked.data,np.exp(1j*t)))
    rgb_array = np.dstack((rgb_array, np.invert(mask)))

    ax.imshow(rgb_array, interpolation="quadric", origin="lower")
    ax.set_axis_off()
x, y = np.mgrid[-2:0.01:2, -2:0.01:2]
r = (x ** 2 + y ** 2) ** 0.5
t = np.arctan2(x, y)
del x, y
if rotation is not None:
    t += math.radians(rotation)
    t = (t + np.pi) % (2 * np.pi) - np.pi
rotation = None
x, y = np.mgrid[-2:0.01:2, -2:0.01:2]
r = (x ** 2 + y ** 2) ** 0.5
t = np.arctan2(x, y)
del x, y
if rotation is not None:
    t += math.radians(rotation)
    t = (t + np.pi) % (2 * np.pi) - np.pi
plt.imshow(t)
rotation = None
x, y = np.mgrid[-2:0.01:2, -2:0.01:2]
r = (x ** 2 + y ** 2) ** 0.5
t = np.arctan2(x, y)
del x, y
# if rotation is not None:
#     t += math.radians(rotation)
#     t = (t + np.pi) % (2 * np.pi) - np.pi
plt.imshow(t)
rotation = None
x, y = np.mgrid[-2:0.01:2, -2:0.01:2]
r = (x ** 2 + y ** 2) ** 0.5
t = np.arctan2(x, y)
del x, y
# if rotation is not None:
#     t += math.radians(rotation)
#     t = (t + np.pi) % (2 * np.pi) - np.pi
plt.imshow(t)
plt.imshow(r)
rotation = None
x, y = np.mgrid[-500:2:500, -500:2:500]
r = (x ** 2 + y ** 2) ** 0.5
t = np.arctan2(x, y)
del x, y
# if rotation is not None:
#     t += math.radians(rotation)
#     t = (t + np.pi) % (2 * np.pi) - np.pi
plt.imshow(r)
rotation = None
x, y = np.mgrid[-500:2:500, -500:2:500]
r = (x ** 2 + y ** 2) ** 0.5
t = np.arctan2(x, y)
del x, y
# if rotation is not None:
#     t += math.radians(rotation)
#     t = (t + np.pi) % (2 * np.pi) - np.pi
plt.imshow(r)
r
x
rotation = None
x, y = np.mgrid[-500:2:500, -500:2:500]
r = (x ** 2 + y ** 2) ** 0.5
t = np.arctan2(x, y)
# del x, y
# if rotation is not None:
#     t += math.radians(rotation)
#     t = (t + np.pi) % (2 * np.pi) - np.pi
plt.imshow(r)
x
rotation = None
x, y = np.linspace(0,2,num=100)
r = (x ** 2 + y ** 2) ** 0.5
t = np.arctan2(x, y)
# del x, y
# if rotation is not None:
#     t += math.radians(rotation)
#     t = (t + np.pi) % (2 * np.pi) - np.pi
rotation = None
x, y = np.linspace(-2,2,num=100),np.linspace(-2,2,num=100)
r = (x ** 2 + y ** 2) ** 0.5
t = np.arctan2(x, y)
# del x, y
# if rotation is not None:
#     t += math.radians(rotation)
#     t = (t + np.pi) % (2 * np.pi) - np.pi
plt.imshow(r)
x
x**2
x**2+y**2
sqrt(x**2+y**2)
(x**2+y**2)**0.5
r = (x**2+y**2)**0.5
x, y = np.mgrid[-2.0:2.0:500j, -2.0:2.0:500j]
x,y
x, y = np.mgrid[-2.0:2.0:500j, -2.0:2.0:500j]
x,y
x, y = np.mgrid[-2.0:2.0:500j, -2.0:2.0:500j]
x,y
r = (x ** 2 + y ** 2) ** 0.5
r
plt.imshow(r)
plt.imshow(t)
plt.imshow(r)
x, y = np.mgrid[-2.0:2.0:500j, -2.0:2.0:500j]
x,y
r = (x ** 2 + y ** 2) ** 0.5
t = np.arctan2(x, y)
plt.imshow(t)
x, y = np.mgrid[-2.0:2.0:500j, -2.0:2.0:500j]
x,y
r = (x ** 2 + y ** 2) ** 0.5
t = np.arctan2(x, y)
t += math.radians(rotation)
t = (t + np.pi) % (2 * np.pi) - np.pi
x, y = np.mgrid[-2.0:2.0:500j, -2.0:2.0:500j]
x,y
r = (x ** 2 + y ** 2) ** 0.5
t = np.arctan2(x, y)
# t += math.radians(rotation)
t = (t + np.pi) % (2 * np.pi) - np.pi
plt.imshow(t)
plt.imshow(np.dot(r,np.exp(t))
plt.imshow(np.dot(r,np.exp(t)))
np.dot(r,np.exp(t))
np.dot(r,1j*np.exp(t))
from abtem.visualize.utils import domain_coloring
def plot_colors(
                array,
                # extent,
                add_colorbars: bool = True,
                num_abs = 5
                ):
    from cplot._colors import get_srgb1,scale01
    from matplotlib import colors,cm
    plt.imshow(
        domain_coloring(array),
        # extent=extent,
        interpolation="nearest",
        origin="lower",
        aspect="equal",
    )

    if add_colorbars:
        # abs colorbar
        norm = colors.Normalize(vmin=np.min(np.abs(array)), vmax=np.max(np.abs(array)))
        cb0 = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cm.gray))
        cb0.set_label("abs", rotation=0, ha="center", va="top")
        cb0.ax.yaxis.set_label_coords(0.5, -0.03)
        abs_sampling = np.linspace(np.min(np.abs(array)),np.max(np.abs(array)),num=num_abs)
        str_abs_sampling = ["{:.2E}".format(n) for n in abs_sampling]
        scaled_vals = scale01(abs_sampling,abs_scaling)
        cb0.set_ticks([0.0, *scaled_vals, 1.0])
        cb0.set_ticklabels(str_abs_sampling)

        # arg colorbar
        # create new colormap
        z = np.exp(1j * np.linspace(-np.pi, np.pi, 256))
        rgb_vals = get_srgb1(z, abs_scaling=abs_scaling, colorspace=colorspace)
        rgba_vals = np.pad(rgb_vals, ((0, 0), (0, 1)), constant_values=1.0)
        newcmp = colors.ListedColormap(rgba_vals)
        #
        norm = colors.Normalize(vmin=-np.pi, vmax=np.pi)
        cb1 = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=newcmp))
        cb1.set_label("arg", rotation=0, ha="center", va="top")
        cb1.ax.yaxis.set_label_coords(0.5, -0.03)
        cb1.set_ticks([-np.pi, -np.pi / 2, 0, +np.pi / 2, np.pi])
        cb1.set_ticklabels(["-π", "-π/2", "0", "π/2", "π"])

from pyxem.utils.pixelated_stem_tools import _get_rgb_phase_magnitude_array
def make_color_wheel(ax, rotation=None):
    x, y = np.mgrid[-2.0:2.0:500j, -2.0:2.0:500j]
    r = (x ** 2 + y ** 2) ** 0.5
    t = np.arctan2(x, y)
    del x, y
    if rotation is not None:
        t += math.radians(rotation)
        t = (t + np.pi) % (2 * np.pi) - np.pi

    r_masked = np.ma.masked_where((2.0 < r) | (r < 1.0), r)
    r_masked -= 1.0

    mask = r_masked.mask
    r_masked.data[r_masked.mask] = r_masked.mean()
    rgb_array = domain_coloring(np.dot(r_masked.data,np.exp(1j*t)))
    rgb_array = np.dstack((rgb_array, np.invert(mask)))

    ax.imshow(rgb_array, interpolation="quadric", origin="lower")
    ax.set_axis_off()
fig, ax = plt.subplots(1, 1, figsize=(7, 7))
ax_indicator = fig.add_subplot(331)
make_color_wheel(ax_indicator)
plt.show()
plt.imshow(abs(np.dot(r,1j*np.exp(t))))
plt.imshow(abs(np.dot(r,np.exp(1j*t))))
plt.imshow(t)
plt.imshow(t)
plt.colorbar
plt.imshow(t)
plt.colorbar()
from abtem.visualize.utils import domain_coloring
def plot_colors(
                array,
                # extent,
                add_colorbars: bool = True,
                num_abs = 5
                ):
    from cplot._colors import get_srgb1,scale01
    from matplotlib import colors,cm
    plt.imshow(
        domain_coloring(array),
        # extent=extent,
        interpolation="nearest",
        origin="lower",
        aspect="equal",
    )

    if add_colorbars:
        # abs colorbar
        norm = colors.Normalize(vmin=np.min(np.abs(array)), vmax=np.max(np.abs(array)))
        cb0 = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cm.gray))
        cb0.set_label("abs", rotation=0, ha="center", va="top")
        cb0.ax.yaxis.set_label_coords(0.5, -0.03)
        abs_sampling = np.linspace(np.min(np.abs(array)),np.max(np.abs(array)),num=num_abs)
        str_abs_sampling = ["{:.2E}".format(n) for n in abs_sampling]
        scaled_vals = scale01(abs_sampling,abs_scaling)
        cb0.set_ticks([0.0, *scaled_vals, 1.0])
        cb0.set_ticklabels(str_abs_sampling)

        # arg colorbar
        # create new colormap
        z = np.exp(1j * np.linspace(-np.pi, np.pi, 256))
        rgb_vals = domain_coloring(z)
        rgba_vals = np.pad(rgb_vals, ((0, 0), (0, 1)), constant_values=1.0)
        newcmp = colors.ListedColormap(rgba_vals)
        #
        norm = colors.Normalize(vmin=-np.pi, vmax=np.pi)
        cb1 = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=newcmp))
        cb1.set_label("arg", rotation=0, ha="center", va="top")
        cb1.ax.yaxis.set_label_coords(0.5, -0.03)
        cb1.set_ticks([-np.pi, -np.pi / 2, 0, +np.pi / 2, np.pi])
        cb1.set_ticklabels(["-π", "-π/2", "0", "π/2", "π"])

from pyxem.utils.pixelated_stem_tools import _get_rgb_phase_magnitude_array
def make_color_wheel(ax, rotation=None):
    x, y = np.mgrid[-2.0:2.0:500j, -2.0:2.0:500j]
    r = (x ** 2 + y ** 2) ** 0.5
    t = np.arctan2(x, y)
    del x, y
    if rotation is not None:
        t += math.radians(rotation)
        t = (t + np.pi) % (2 * np.pi) - np.pi

    r_masked = np.ma.masked_where((2.0 < r) | (r < 1.0), r)
    r_masked -= 1.0

    mask = r_masked.mask
    r_masked.data[r_masked.mask] = r_masked.mean()
    rgb_array = domain_coloring(np.dot(r_masked.data,np.exp(1j*t)))
    rgb_array = np.dstack((rgb_array, np.invert(mask)))

    ax.imshow(rgb_array, interpolation="quadric", origin="lower")
    ax.set_axis_off()
fig, ax = plt.subplots(1, 1, figsize=(7, 7))
ax_indicator = fig.add_subplot(331)
make_color_wheel(ax_indicator)
plt.show()
plot_colors(atomic_transition_potentials[1].measure().array,add_colorbars=True)
from abtem.visualize.utils import domain_coloring
def plot_colors(
                array,
                # extent,
                add_colorbars: bool = True,
                num_abs = 5
                ):
    from cplot._colors import get_srgb1,scale01
    from matplotlib import colors,cm
    plt.imshow(
        domain_coloring(array),
        # extent=extent,
        interpolation="nearest",
        origin="lower",
        aspect="equal",
    )

    if add_colorbars:
        # abs colorbar
        norm = colors.Normalize(vmin=np.min(np.abs(array)), vmax=np.max(np.abs(array)))
        cb0 = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cm.gray))
        cb0.set_label("abs", rotation=0, ha="center", va="top")
        cb0.ax.yaxis.set_label_coords(0.5, -0.03)
        abs_sampling = np.linspace(np.min(np.abs(array)),np.max(np.abs(array)),num=num_abs)
        str_abs_sampling = ["{:.2E}".format(n) for n in abs_sampling]
        scaled_vals = abs_sampling/np.abs(array).ptp
        cb0.set_ticks([0.0, *scaled_vals, 1.0])
        cb0.set_ticklabels(str_abs_sampling)

        # arg colorbar
        # create new colormap
        z = np.exp(1j * np.linspace(-np.pi, np.pi, 256))
        rgb_vals = domain_coloring(z)
        rgba_vals = np.pad(rgb_vals, ((0, 0), (0, 1)), constant_values=1.0)
        newcmp = colors.ListedColormap(rgba_vals)
        #
        norm = colors.Normalize(vmin=-np.pi, vmax=np.pi)
        cb1 = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=newcmp))
        cb1.set_label("arg", rotation=0, ha="center", va="top")
        cb1.ax.yaxis.set_label_coords(0.5, -0.03)
        cb1.set_ticks([-np.pi, -np.pi / 2, 0, +np.pi / 2, np.pi])
        cb1.set_ticklabels(["-π", "-π/2", "0", "π/2", "π"])

from pyxem.utils.pixelated_stem_tools import _get_rgb_phase_magnitude_array
def make_color_wheel(ax, rotation=None):
    x, y = np.mgrid[-2.0:2.0:500j, -2.0:2.0:500j]
    r = (x ** 2 + y ** 2) ** 0.5
    t = np.arctan2(x, y)
    del x, y
    if rotation is not None:
        t += math.radians(rotation)
        t = (t + np.pi) % (2 * np.pi) - np.pi

    r_masked = np.ma.masked_where((2.0 < r) | (r < 1.0), r)
    r_masked -= 1.0

    mask = r_masked.mask
    r_masked.data[r_masked.mask] = r_masked.mean()
    rgb_array = domain_coloring(np.dot(r_masked.data,np.exp(1j*t)))
    rgb_array = np.dstack((rgb_array, np.invert(mask)))

    ax.imshow(rgb_array, interpolation="quadric", origin="lower")
    ax.set_axis_off()
fig, ax = plt.subplots(1, 1, figsize=(7, 7))
ax_indicator = fig.add_subplot(331)
make_color_wheel(ax_indicator)
plt.show()
plot_colors(atomic_transition_potentials[1].measure().array,add_colorbars=True)
from abtem.visualize.utils import domain_coloring
def plot_colors(
                array,
                # extent,
                add_colorbars: bool = True,
                num_abs = 5
                ):
    from cplot._colors import get_srgb1,scale01
    from matplotlib import colors,cm
    plt.imshow(
        domain_coloring(array),
        # extent=extent,
        interpolation="nearest",
        origin="lower",
        aspect="equal",
    )

    if add_colorbars:
        # abs colorbar
        norm = colors.Normalize(vmin=np.min(np.abs(array)), vmax=np.max(np.abs(array)))
        cb0 = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cm.gray))
        cb0.set_label("abs", rotation=0, ha="center", va="top")
        cb0.ax.yaxis.set_label_coords(0.5, -0.03)
        abs_sampling = np.linspace(np.min(np.abs(array)),np.max(np.abs(array)),num=num_abs)
        str_abs_sampling = ["{:.2E}".format(n) for n in abs_sampling]
        scaled_vals = abs_sampling/np.ptp(np.abs(array))
        cb0.set_ticks([0.0, *scaled_vals, 1.0])
        cb0.set_ticklabels(str_abs_sampling)

        # arg colorbar
        # create new colormap
        z = np.exp(1j * np.linspace(-np.pi, np.pi, 256))
        rgb_vals = domain_coloring(z)
        rgba_vals = np.pad(rgb_vals, ((0, 0), (0, 1)), constant_values=1.0)
        newcmp = colors.ListedColormap(rgba_vals)
        #
        norm = colors.Normalize(vmin=-np.pi, vmax=np.pi)
        cb1 = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=newcmp))
        cb1.set_label("arg", rotation=0, ha="center", va="top")
        cb1.ax.yaxis.set_label_coords(0.5, -0.03)
        cb1.set_ticks([-np.pi, -np.pi / 2, 0, +np.pi / 2, np.pi])
        cb1.set_ticklabels(["-π", "-π/2", "0", "π/2", "π"])

from pyxem.utils.pixelated_stem_tools import _get_rgb_phase_magnitude_array
def make_color_wheel(ax, rotation=None):
    x, y = np.mgrid[-2.0:2.0:500j, -2.0:2.0:500j]
    r = (x ** 2 + y ** 2) ** 0.5
    t = np.arctan2(x, y)
    del x, y
    if rotation is not None:
        t += math.radians(rotation)
        t = (t + np.pi) % (2 * np.pi) - np.pi

    r_masked = np.ma.masked_where((2.0 < r) | (r < 1.0), r)
    r_masked -= 1.0

    mask = r_masked.mask
    r_masked.data[r_masked.mask] = r_masked.mean()
    rgb_array = domain_coloring(np.dot(r_masked.data,np.exp(1j*t)))
    rgb_array = np.dstack((rgb_array, np.invert(mask)))

    ax.imshow(rgb_array, interpolation="quadric", origin="lower")
    ax.set_axis_off()
fig, ax = plt.subplots(1, 1, figsize=(7, 7))
ax_indicator = fig.add_subplot(331)
make_color_wheel(ax_indicator)
plt.show()
plot_colors(atomic_transition_potentials[1].measure().array,add_colorbars=True)
import debugpy
debugpy.debug_this_thread()
plot_colors(atomic_transition_potentials[1].measure().array,add_colorbars=True)
from abtem.visualize.utils import domain_coloring
def plot_colors(
                array,
                # extent,
                add_colorbars: bool = True,
                num_abs = 5
                ):
    from cplot._colors import get_srgb1,scale01
    from matplotlib import colors,cm
    plt.imshow(
        domain_coloring(array),
        # extent=extent,
        interpolation="nearest",
        origin="lower",
        aspect="equal",
    )

    if add_colorbars:
        # abs colorbar
        norm = colors.Normalize(vmin=np.min(np.abs(array)), vmax=np.max(np.abs(array)))
        cb0 = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cm.gray))
        cb0.set_label("abs", rotation=0, ha="center", va="top")
        cb0.ax.yaxis.set_label_coords(0.5, -0.03)
        abs_sampling = np.linspace(np.min(np.abs(array)),np.max(np.abs(array)),num=num_abs)
        str_abs_sampling = ["{:.2E}".format(n) for n in abs_sampling]
        scaled_vals = scale01(abs_sampling,abs_scaling)
        cb0.set_ticks([0.0, *scaled_vals, 1.0])
        cb0.set_ticklabels(str_abs_sampling)

        # arg colorbar
        # create new colormap
        z = np.exp(1j * np.linspace(-np.pi, np.pi, 256))
        rgb_vals = get_srgb1(z, abs_scaling=abs_scaling, colorspace=colorspace)
        rgba_vals = np.pad(rgb_vals, ((0, 0), (0, 1)), constant_values=1.0)
        newcmp = colors.ListedColormap(rgba_vals)
        #
        norm = colors.Normalize(vmin=-np.pi, vmax=np.pi)
        cb1 = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=newcmp))
        cb1.set_label("arg", rotation=0, ha="center", va="top")
        cb1.ax.yaxis.set_label_coords(0.5, -0.03)
        cb1.set_ticks([-np.pi, -np.pi / 2, 0, +np.pi / 2, np.pi])
        cb1.set_ticklabels(["-π", "-π/2", "0", "π/2", "π"])

plot_colors(atomic_transition_potentials[1].measure().array,add_colorbars=True)
from abtem.visualize.utils import domain_coloring
def plot_colors(
                array,
                # extent,
                add_colorbars: bool = True,
                num_abs = 5
                ):
    from cplot._colors import get_srgb1,scale01
    from matplotlib import colors,cm
    abs_scaling = "h-1.0",
    colorspace = "cam16",
    plt.imshow(
        domain_coloring(array),
        # extent=extent,
        interpolation="nearest",
        origin="lower",
        aspect="equal",
    )

    if add_colorbars:
        # abs colorbar
        norm = colors.Normalize(vmin=np.min(np.abs(array)), vmax=np.max(np.abs(array)))
        cb0 = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cm.gray))
        cb0.set_label("abs", rotation=0, ha="center", va="top")
        cb0.ax.yaxis.set_label_coords(0.5, -0.03)
        abs_sampling = np.linspace(np.min(np.abs(array)),np.max(np.abs(array)),num=num_abs)
        str_abs_sampling = ["{:.2E}".format(n) for n in abs_sampling]
        scaled_vals = scale01(abs_sampling,abs_scaling)
        cb0.set_ticks([0.0, *scaled_vals, 1.0])
        cb0.set_ticklabels(str_abs_sampling)

        # arg colorbar
        # create new colormap
        z = np.exp(1j * np.linspace(-np.pi, np.pi, 256))
        rgb_vals = get_srgb1(z, abs_scaling=abs_scaling, colorspace=colorspace)
        rgba_vals = np.pad(rgb_vals, ((0, 0), (0, 1)), constant_values=1.0)
        newcmp = colors.ListedColormap(rgba_vals)
        #
        norm = colors.Normalize(vmin=-np.pi, vmax=np.pi)
        cb1 = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=newcmp))
        cb1.set_label("arg", rotation=0, ha="center", va="top")
        cb1.ax.yaxis.set_label_coords(0.5, -0.03)
        cb1.set_ticks([-np.pi, -np.pi / 2, 0, +np.pi / 2, np.pi])
        cb1.set_ticklabels(["-π", "-π/2", "0", "π/2", "π"])

plot_colors(atomic_transition_potentials[1].measure().array,add_colorbars=True)
from abtem.visualize.utils import domain_coloring
def plot_colors(
                array,
                # extent,
                add_colorbars: bool = True,
                num_abs = 5
                ):
    from cplot._colors import get_srgb1,scale01
    from matplotlib import colors,cm
    
    plt.imshow(
        domain_coloring(array),
        # extent=extent,
        interpolation="nearest",
        origin="lower",
        aspect="equal",
    )

    if add_colorbars:
        # abs colorbar
        vmin = np.min(np.abs(array))
        vmax = np.max(np.abs(array))
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
        cb0 = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cm.gray))
        cb0.set_label("abs", rotation=0, ha="center", va="top")
        cb0.ax.yaxis.set_label_coords(0.5, -0.03)
        abs_sampling = np.linspace(vmin,vmax,num=num_abs)
        str_abs_sampling = ["{:.2E}".format(n) for n in abs_sampling]
        scaled_vals = (abs_sampling - vmin)/np.ptp(array)
        cb0.set_ticks([0.0, *scaled_vals, 1.0])
        cb0.set_ticklabels(str_abs_sampling)

        # arg colorbar
        # create new colormap
        z = np.exp(1j * np.linspace(-np.pi, np.pi, 256))
        rgb_vals = get_srgb1(z, abs_scaling=abs_scaling, colorspace=colorspace)
        rgba_vals = np.pad(rgb_vals, ((0, 0), (0, 1)), constant_values=1.0)
        newcmp = colors.ListedColormap(rgba_vals)
        #
        norm = colors.Normalize(vmin=-np.pi, vmax=np.pi)
        cb1 = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=newcmp))
        cb1.set_label("arg", rotation=0, ha="center", va="top")
        cb1.ax.yaxis.set_label_coords(0.5, -0.03)
        cb1.set_ticks([-np.pi, -np.pi / 2, 0, +np.pi / 2, np.pi])
        cb1.set_ticklabels(["-π", "-π/2", "0", "π/2", "π"])

from pyxem.utils.pixelated_stem_tools import _get_rgb_phase_magnitude_array
def make_color_wheel(ax, rotation=None):
    x, y = np.mgrid[-2.0:2.0:500j, -2.0:2.0:500j]
    r = (x ** 2 + y ** 2) ** 0.5
    t = np.arctan2(x, y)
    del x, y
    if rotation is not None:
        t += math.radians(rotation)
        t = (t + np.pi) % (2 * np.pi) - np.pi

    r_masked = np.ma.masked_where((2.0 < r) | (r < 1.0), r)
    r_masked -= 1.0

    mask = r_masked.mask
    r_masked.data[r_masked.mask] = r_masked.mean()
    rgb_array = domain_coloring(np.dot(r_masked))
    rgb_array = np.dstack((rgb_array, np.invert(mask)))

    ax.imshow(rgb_array, interpolation="quadric", origin="lower")
    ax.set_axis_off()
fig, ax = plt.subplots(1, 1, figsize=(7, 7))
ax_indicator = fig.add_subplot(331)
make_color_wheel(ax_indicator)
plt.show()
plot_colors(atomic_transition_potentials[1].measure().array,add_colorbars=True)
from abtem.visualize.utils import domain_coloring
def plot_colors(
                array,
                # extent,
                add_colorbars: bool = True,
                num_abs = 5
                ):
    from cplot._colors import get_srgb1,scale01
    from matplotlib import colors,cm
    
    plt.imshow(
        domain_coloring(array),
        # extent=extent,
        interpolation="nearest",
        origin="lower",
        aspect="equal",
    )

    if add_colorbars:
        # abs colorbar
        vmin = np.min(np.abs(array))
        vmax = np.max(np.abs(array))
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
        cb0 = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cm.gray))
        cb0.set_label("abs", rotation=0, ha="center", va="top")
        cb0.ax.yaxis.set_label_coords(0.5, -0.03)
        abs_sampling = np.linspace(vmin,vmax,num=num_abs)
        str_abs_sampling = ["{:.2E}".format(n) for n in abs_sampling]
        scaled_vals = (abs_sampling - vmin)/np.ptp(array)
        cb0.set_ticks([0.0, *scaled_vals, 1.0])
        cb0.set_ticklabels(str_abs_sampling)

        # arg colorbar
        # create new colormap
        z = np.exp(1j * np.linspace(-np.pi, np.pi, 512))
        rgb_vals = domain_coloring(z)
        rgba_vals = np.pad(rgb_vals, ((0, 0), (0, 1)), constant_values=1.0)
        newcmp = colors.ListedColormap(rgba_vals)
        #
        norm = colors.Normalize(vmin=-np.pi, vmax=np.pi)
        cb1 = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=newcmp))
        cb1.set_label("arg", rotation=0, ha="center", va="top")
        cb1.ax.yaxis.set_label_coords(0.5, -0.03)
        cb1.set_ticks([-np.pi, -np.pi / 2, 0, +np.pi / 2, np.pi])
        cb1.set_ticklabels(["-π", "-π/2", "0", "π/2", "π"])

from pyxem.utils.pixelated_stem_tools import _get_rgb_phase_magnitude_array
def make_color_wheel(ax, rotation=None):
    x, y = np.mgrid[-2.0:2.0:500j, -2.0:2.0:500j]
    r = (x ** 2 + y ** 2) ** 0.5
    t = np.arctan2(x, y)
    del x, y
    if rotation is not None:
        t += math.radians(rotation)
        t = (t + np.pi) % (2 * np.pi) - np.pi

    r_masked = np.ma.masked_where((2.0 < r) | (r < 1.0), r)
    r_masked -= 1.0

    mask = r_masked.mask
    r_masked.data[r_masked.mask] = r_masked.mean()
    rgb_array = domain_coloring(np.dot(r_masked))
    rgb_array = np.dstack((rgb_array, np.invert(mask)))

    ax.imshow(rgb_array, interpolation="quadric", origin="lower")
    ax.set_axis_off()
plot_colors(atomic_transition_potentials[1].measure().array,add_colorbars=True)
import debugpy
debugpy.debug_this_thread()
plot_colors(atomic_transition_potentials[1].measure().array,add_colorbars=True)
from abtem.visualize.utils import domain_coloring
def plot_colors(
                array,
                # extent,
                add_colorbars: bool = True,
                num_abs = 5
                ):
    from cplot._colors import get_srgb1,scale01
    from matplotlib import colors,cm
    
    plt.imshow(
        domain_coloring(array),
        # extent=extent,
        interpolation="nearest",
        origin="lower",
        aspect="equal",
    )

    if add_colorbars:
        # abs colorbar
        vmin = np.min(np.abs(array))
        vmax = np.max(np.abs(array))
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
        cb0 = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cm.gray))
        cb0.set_label("abs", rotation=0, ha="center", va="top")
        cb0.ax.yaxis.set_label_coords(0.5, -0.03)
        abs_sampling = np.linspace(vmin,vmax,num=num_abs)
        str_abs_sampling = ["{:.2E}".format(n) for n in abs_sampling]
        scaled_vals = (abs_sampling - vmin)/np.ptp(np.abs(array))
        cb0.set_ticks([0.0, *scaled_vals, 1.0])
        cb0.set_ticklabels(str_abs_sampling)

        # arg colorbar
        # create new colormap
        z = np.exp(1j * np.linspace(-np.pi, np.pi, 512))
        rgb_vals = domain_coloring(z)
        rgba_vals = np.pad(rgb_vals, ((0, 0), (0, 1)), constant_values=1.0)
        newcmp = colors.ListedColormap(rgba_vals)
        #
        norm = colors.Normalize(vmin=-np.pi, vmax=np.pi)
        cb1 = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=newcmp))
        cb1.set_label("arg", rotation=0, ha="center", va="top")
        cb1.ax.yaxis.set_label_coords(0.5, -0.03)
        cb1.set_ticks([-np.pi, -np.pi / 2, 0, +np.pi / 2, np.pi])
        cb1.set_ticklabels(["-π", "-π/2", "0", "π/2", "π"])

plot_colors(atomic_transition_potentials[1].measure().array,add_colorbars=True)
import debugpy
debugpy.debug_this_thread()
plot_colors(atomic_transition_potentials[1].measure().array,add_colorbars=True)
from abtem.visualize.utils import domain_coloring
def plot_colors(
                array,
                # extent,
                add_colorbars: bool = True,
                num_abs = 5
                ):
    from cplot._colors import get_srgb1,scale01
    from matplotlib import colors,cm
    
    plt.imshow(
        domain_coloring(array),
        # extent=extent,
        interpolation="nearest",
        origin="lower",
        aspect="equal",
    )

    if add_colorbars:
        # abs colorbar
        vmin = np.min(np.abs(array))
        vmax = np.max(np.abs(array))
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
        cb0 = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cm.gray))
        cb0.set_label("abs", rotation=0, ha="center", va="top")
        cb0.ax.yaxis.set_label_coords(0.5, -0.03)
        abs_sampling = np.linspace(vmin,vmax,num=num_abs)
        str_abs_sampling = ["{:.1E}".format(n) for n in abs_sampling]
        scaled_vals = (abs_sampling - vmin)/np.ptp(np.abs(array))
        cb0.set_ticks([0.0, *scaled_vals, 1.0])
        cb0.set_ticklabels(str_abs_sampling)

        # arg colorbar
        # create new colormap
        z = np.exp(1j * np.linspace(-np.pi, np.pi, 512))
        rgb_vals = domain_coloring(z)
        rgba_vals = np.pad(rgb_vals, ((0, 0), (0, 1)), constant_values=1.0)
        newcmp = colors.ListedColormap(rgba_vals)
        #
        norm = colors.Normalize(vmin=-np.pi, vmax=np.pi)
        cb1 = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=newcmp))
        cb1.set_label("arg", rotation=0, ha="center", va="top")
        cb1.ax.yaxis.set_label_coords(0.5, -0.03)
        cb1.set_ticks([-np.pi, -np.pi / 2, 0, +np.pi / 2, np.pi])
        cb1.set_ticklabels(["-π", "-π/2", "0", "π/2", "π"])

plot_colors(atomic_transition_potentials[1].measure().array,add_colorbars=True)
from abtem.visualize.utils import domain_coloring
def plot_colors(
                array,
                # extent,
                add_colorbars: bool = True,
                num_abs = 5
                ):
    from cplot._colors import get_srgb1,scale01
    from matplotlib import colors,cm
    
    plt.imshow(
        domain_coloring(array),
        # extent=extent,
        interpolation="nearest",
        origin="lower",
        aspect="equal",
    )

    if add_colorbars:
        # abs colorbar
        vmin = np.min(np.abs(array))
        vmax = np.max(np.abs(array))
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
        cb0 = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cm.gray))
        cb0.set_label("abs", rotation=0, ha="center", va="top")
        cb0.ax.yaxis.set_label_coords(0.5, -0.03)
        abs_sampling = np.linspace(vmin,vmax,num=num_abs)
        str_abs_sampling = ["{:.1E}".format(n) for n in abs_sampling]
        scaled_vals = (abs_sampling - vmin)/np.ptp(np.abs(array))
        cb0.set_ticks([0.0, *scaled_vals, 1.0])
        cb0.set_ticklabels(str_abs_sampling)

        # arg colorbar
        # create new colormap
        # z = np.exp(1j * np.linspace(-np.pi, np.pi, 512))
        # rgb_vals = domain_coloring(z)
        # rgba_vals = np.pad(rgb_vals, ((0, 0), (0, 1)), constant_values=1.0)
        # newcmp = colors.ListedColormap(rgba_vals)
        # #
        # norm = colors.Normalize(vmin=-np.pi, vmax=np.pi)
        # cb1 = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=newcmp))
        # cb1.set_label("arg", rotation=0, ha="center", va="top")
        # cb1.ax.yaxis.set_label_coords(0.5, -0.03)
        # cb1.set_ticks([-np.pi, -np.pi / 2, 0, +np.pi / 2, np.pi])
        # cb1.set_ticklabels(["-π", "-π/2", "0", "π/2", "π"])

plot_colors(atomic_transition_potentials[1].measure().array,add_colorbars=True)
from abtem.visualize.utils import domain_coloring
def plot_colors(
                array,
                # extent,
                add_colorbars: bool = True,
                num_abs = 5
                ):
    from cplot._colors import get_srgb1,scale01
    from matplotlib import colors,cm
    
    plt.imshow(
        domain_coloring(array),
        # extent=extent,
        interpolation="nearest",
        origin="lower",
        aspect="equal",
    )

    if add_colorbars:
        # abs colorbar
        vmin = np.min(np.abs(array))
        vmax = np.max(np.abs(array))
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
        cb0 = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cm.gray))
        cb0.set_label("abs", rotation=0, ha="center", va="top")
        cb0.ax.xaxis.set_label_coords(0.5, -0.03)
        abs_sampling = np.linspace(vmin,vmax,num=num_abs)
        str_abs_sampling = ["{:.1E}".format(n) for n in abs_sampling]
        scaled_vals = (abs_sampling - vmin)/np.ptp(np.abs(array))
        cb0.set_ticks([0.0, *scaled_vals, 1.0])
        cb0.set_ticklabels(str_abs_sampling)

        # arg colorbar
        # create new colormap
        # z = np.exp(1j * np.linspace(-np.pi, np.pi, 512))
        # rgb_vals = domain_coloring(z)
        # rgba_vals = np.pad(rgb_vals, ((0, 0), (0, 1)), constant_values=1.0)
        # newcmp = colors.ListedColormap(rgba_vals)
        # #
        # norm = colors.Normalize(vmin=-np.pi, vmax=np.pi)
        # cb1 = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=newcmp))
        # cb1.set_label("arg", rotation=0, ha="center", va="top")
        # cb1.ax.yaxis.set_label_coords(0.5, -0.03)
        # cb1.set_ticks([-np.pi, -np.pi / 2, 0, +np.pi / 2, np.pi])
        # cb1.set_ticklabels(["-π", "-π/2", "0", "π/2", "π"])

plot_colors(atomic_transition_potentials[1].measure().array,add_colorbars=True)
from abtem.visualize.utils import domain_coloring
def plot_colors(
                array,
                # extent,
                add_colorbars: bool = True,
                num_abs = 5
                ):
    from cplot._colors import get_srgb1,scale01
    from matplotlib import colors,cm
    
    plt.imshow(
        domain_coloring(array),
        # extent=extent,
        interpolation="nearest",
        origin="lower",
        aspect="equal",
    )

    if add_colorbars:
        # abs colorbar
        vmin = np.min(np.abs(array))
        vmax = np.max(np.abs(array))
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
        cb0 = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cm.gray))
        cb0.set_label("abs", rotation=0, ha="center", va="top")
        cb0.ax.yaxis.set_label_coords(0.5, -0.03)
        abs_sampling = np.linspace(vmin,vmax,num=num_abs)
        str_abs_sampling = ["{:.1E}".format(n) for n in abs_sampling]
        scaled_vals = (abs_sampling - vmin)/np.ptp(np.abs(array))
        cb0.set_ticks([0.0, *scaled_vals, 1.0])
        cb0.set_ticklabels(str_abs_sampling)

        # arg colorbar
        # create new colormap
        # z = np.exp(1j * np.linspace(-np.pi, np.pi, 512))
        # rgb_vals = domain_coloring(z)
        # rgba_vals = np.pad(rgb_vals, ((0, 0), (0, 1)), constant_values=1.0)
        # newcmp = colors.ListedColormap(rgba_vals)
        # #
        # norm = colors.Normalize(vmin=-np.pi, vmax=np.pi)
        # cb1 = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=newcmp))
        # cb1.set_label("arg", rotation=0, ha="center", va="top")
        # cb1.ax.yaxis.set_label_coords(0.5, -0.03)
        # cb1.set_ticks([-np.pi, -np.pi / 2, 0, +np.pi / 2, np.pi])
        # cb1.set_ticklabels(["-π", "-π/2", "0", "π/2", "π"])

import debugpy
debugpy.debug_this_thread()
plot_colors(atomic_transition_potentials[1].measure().array,add_colorbars=True)
from abtem.visualize.utils import domain_coloring
def plot_colors(
                array,
                # extent,
                add_colorbars: bool = True,
                num_abs = 5
                ):
    from cplot._colors import get_srgb1,scale01
    from matplotlib import colors,cm
    
    plt.imshow(
        domain_coloring(array),
        # extent=extent,
        interpolation="nearest",
        origin="lower",
        aspect="equal",
    )

    if add_colorbars:
        # abs colorbar
        vmin = np.min(np.abs(array))
        vmax = np.max(np.abs(array))
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
        cb0 = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cm.gray))
        cb0.set_label("abs", rotation=0, ha="center", va="top")
        cb0.ax.yaxis.set_label_coords(0.5, -0.03)
        abs_sampling = np.linspace(vmin,vmax,num=num_abs)
        str_abs_sampling = ["{:.1E}".format(n) for n in abs_sampling]
        scaled_vals = (abs_sampling - vmin)/np.ptp(np.abs(array))
        cb0.set_ticks([0.0, scaled_vals, 1.0])
        cb0.set_ticklabels(str_abs_sampling)

        # arg colorbar
        # create new colormap
        # z = np.exp(1j * np.linspace(-np.pi, np.pi, 512))
        # rgb_vals = domain_coloring(z)
        # rgba_vals = np.pad(rgb_vals, ((0, 0), (0, 1)), constant_values=1.0)
        # newcmp = colors.ListedColormap(rgba_vals)
        # #
        # norm = colors.Normalize(vmin=-np.pi, vmax=np.pi)
        # cb1 = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=newcmp))
        # cb1.set_label("arg", rotation=0, ha="center", va="top")
        # cb1.ax.yaxis.set_label_coords(0.5, -0.03)
        # cb1.set_ticks([-np.pi, -np.pi / 2, 0, +np.pi / 2, np.pi])
        # cb1.set_ticklabels(["-π", "-π/2", "0", "π/2", "π"])

plot_colors(atomic_transition_potentials[1].measure().array,add_colorbars=True)
import debugpy
debugpy.debug_this_thread()
plot_colors(atomic_transition_potentials[1].measure().array,add_colorbars=True)
from abtem.visualize.utils import domain_coloring
def plot_colors(
                array,
                # extent,
                add_colorbars: bool = True,
                num_abs = 5
                ):
    from cplot._colors import get_srgb1,scale01
    from matplotlib import colors,cm
    
    plt.imshow(
        domain_coloring(array),
        # extent=extent,
        interpolation="nearest",
        origin="lower",
        aspect="equal",
    )

    if add_colorbars:
        # abs colorbar
        vmin = np.min(np.abs(array))
        vmax = np.max(np.abs(array))
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
        cb0 = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cm.gray))
        cb0.set_label("abs", rotation=0, ha="center", va="top")
        cb0.ax.yaxis.set_label_coords(0.5, -0.03)
        abs_sampling = np.linspace(vmin,vmax,num=num_abs)
        str_abs_sampling = ["{:.1E}".format(n) for n in abs_sampling]
        scaled_vals = (abs_sampling - vmin)/np.ptp(np.abs(array))
        cb0.set_ticks(scaled_vals.tolist())
        cb0.set_ticklabels(str_abs_sampling)

        # arg colorbar
        # create new colormap
        # z = np.exp(1j * np.linspace(-np.pi, np.pi, 512))
        # rgb_vals = domain_coloring(z)
        # rgba_vals = np.pad(rgb_vals, ((0, 0), (0, 1)), constant_values=1.0)
        # newcmp = colors.ListedColormap(rgba_vals)
        # #
        # norm = colors.Normalize(vmin=-np.pi, vmax=np.pi)
        # cb1 = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=newcmp))
        # cb1.set_label("arg", rotation=0, ha="center", va="top")
        # cb1.ax.yaxis.set_label_coords(0.5, -0.03)
        # cb1.set_ticks([-np.pi, -np.pi / 2, 0, +np.pi / 2, np.pi])
        # cb1.set_ticklabels(["-π", "-π/2", "0", "π/2", "π"])

plot_colors(atomic_transition_potentials[1].measure().array,add_colorbars=True)
plt.imshow(abs(np.dot(r,np.exp(1j*t))))
ax.set_ticks(scaled_vals.tolist())
fig,ax = plt.subplot(1,1,figsize=(7,7))
plt.imshow(abs(np.dot(r,np.exp(1j*t))))
ax.set_ticks(scaled_vals.tolist())
fig,ax = plt.subplot(111)
plt.imshow(abs(np.dot(r,np.exp(1j*t))))
ax.set_ticks(scaled_vals.tolist())
fig,ax = plt.subplot(1)
plt.imshow(abs(np.dot(r,np.exp(1j*t))))
ax.set_ticks(scaled_vals.tolist())
fig,ax = plt.subplot()
plt.imshow(abs(np.dot(r,np.exp(1j*t))))
ax.set_ticks(scaled_vals.tolist())
fig,ax = plt.subplot(1,1,1)
plt.imshow(abs(np.dot(r,np.exp(1j*t))))
ax.set_ticks(scaled_vals.tolist())
fig,ax = plt.subplot(121)
plt.imshow(abs(np.dot(r,np.exp(1j*t))))
ax.set_ticks(scaled_vals.tolist())
fig,ax = plt.subplot(2,2,1)
plt.imshow(abs(np.dot(r,np.exp(1j*t))))
ax.set_ticks(scaled_vals.tolist())
fig,ax = plt.subplots(1,4, figsize = (20,5))
plt.imshow(abs(np.dot(r,np.exp(1j*t))))
ax.set_ticks(scaled_vals.tolist())
fig,ax = plt.subplots(1,1, figsize = (20,5))
plt.imshow(abs(np.dot(r,np.exp(1j*t))))
ax.set_ticks(scaled_vals.tolist())
fig,ax = plt.subplots(1,1, figsize = (20,5))
plt.imshow(abs(np.dot(r,np.exp(1j*t))))
plt.colorbar()
ax.set_ticks(scaled_vals.tolist())
fig,ax = plt.subplots(1,1, figsize = (20,5))
plt.imshow(abs(np.dot(r,np.exp(1j*t))))
cb=plt.colorbar()
cb.set_ticks(scaled_vals.tolist())
fig,ax = plt.subplots(1,1, figsize = (20,5))
plt.imshow(abs(np.dot(r,np.exp(1j*t))))
cb=plt.colorbar()
cb.set_ticks([900])
from abtem.visualize.utils import domain_coloring
def plot_colors(
                array,
                # extent,
                add_colorbars: bool = True,
                num_abs = 5
                ):
    from cplot._colors import get_srgb1,scale01
    from matplotlib import colors,cm
    
    plt.imshow(
        domain_coloring(array),
        # extent=extent,
        interpolation="nearest",
        origin="lower",
        aspect="equal",
    )

    if add_colorbars:
        # abs colorbar
        vmin = np.min(np.abs(array))
        vmax = np.max(np.abs(array))
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
        cb0 = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cm.gray))
        cb0.set_label("abs", rotation=0, ha="center", va="top")
        cb0.ax.yaxis.set_label_coords(0.5, -0.03)
        abs_sampling = np.linspace(vmin,vmax,num=num_abs)
        str_abs_sampling = ["{:.1E}".format(n) for n in abs_sampling]
        cb0.set_ticks(np.linspace(0,1,num=num_abs))
        cb0.set_ticklabels(str_abs_sampling)

        # arg colorbar
        # create new colormap
        # z = np.exp(1j * np.linspace(-np.pi, np.pi, 512))
        # rgb_vals = domain_coloring(z)
        # rgba_vals = np.pad(rgb_vals, ((0, 0), (0, 1)), constant_values=1.0)
        # newcmp = colors.ListedColormap(rgba_vals)
        # #
        # norm = colors.Normalize(vmin=-np.pi, vmax=np.pi)
        # cb1 = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=newcmp))
        # cb1.set_label("arg", rotation=0, ha="center", va="top")
        # cb1.ax.yaxis.set_label_coords(0.5, -0.03)
        # cb1.set_ticks([-np.pi, -np.pi / 2, 0, +np.pi / 2, np.pi])
        # cb1.set_ticklabels(["-π", "-π/2", "0", "π/2", "π"])

plot_colors(atomic_transition_potentials[1].measure().array,add_colorbars=True)
from abtem.visualize.utils import domain_coloring
def plot_colors(
                array,
                # extent,
                add_colorbars: bool = True,
                num_abs = 5
                ):
    from cplot._colors import get_srgb1,scale01
    from matplotlib import colors,cm
    
    plt.imshow(
        domain_coloring(array),
        # extent=extent,
        interpolation="nearest",
        origin="lower",
        aspect="equal",
    )

    if add_colorbars:
        # abs colorbar
        vmin = np.min(np.abs(array))
        vmax = np.max(np.abs(array))
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
        cb0 = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cm.gray))
        cb0.set_label("abs", rotation=0, ha="center", va="top")
        cb0.ax.yaxis.set_label_coords(0.5, -0.03)
        abs_sampling = np.linspace(vmin,vmax,num=num_abs)
        str_abs_sampling = ["{:.1E}".format(n) for n in abs_sampling]
        # cb0.set_ticks(np.linspace(0,1,num=num_abs))
        # cb0.set_ticklabels(str_abs_sampling)

        # arg colorbar
        # create new colormap
        # z = np.exp(1j * np.linspace(-np.pi, np.pi, 512))
        # rgb_vals = domain_coloring(z)
        # rgba_vals = np.pad(rgb_vals, ((0, 0), (0, 1)), constant_values=1.0)
        # newcmp = colors.ListedColormap(rgba_vals)
        # #
        # norm = colors.Normalize(vmin=-np.pi, vmax=np.pi)
        # cb1 = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=newcmp))
        # cb1.set_label("arg", rotation=0, ha="center", va="top")
        # cb1.ax.yaxis.set_label_coords(0.5, -0.03)
        # cb1.set_ticks([-np.pi, -np.pi / 2, 0, +np.pi / 2, np.pi])
        # cb1.set_ticklabels(["-π", "-π/2", "0", "π/2", "π"])

plot_colors(atomic_transition_potentials[1].measure().array,add_colorbars=True)
from abtem.visualize.utils import domain_coloring
def plot_colors(
                array,
                # extent,
                add_colorbars: bool = True,
                num_abs = 5
                ):
    from cplot._colors import get_srgb1,scale01
    from matplotlib import colors,cm
    
    plt.imshow(
        domain_coloring(array),
        # extent=extent,
        interpolation="nearest",
        origin="lower",
        aspect="equal",
    )

    if add_colorbars:
        # abs colorbar
        vmin = np.min(np.abs(array))
        vmax = np.max(np.abs(array))
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
        cb0 = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cm.gray))
        cb0.set_label("abs", rotation=0, ha="center", va="top")
        cb0.ax.yaxis.set_label_coords(0.5, -0.03)
        # abs_sampling = np.linspace(vmin,vmax,num=num_abs)
        # str_abs_sampling = ["{:.1E}".format(n) for n in abs_sampling]
        # cb0.set_ticks(np.linspace(0,1,num=num_abs))
        # cb0.set_ticklabels(str_abs_sampling)

        # arg colorbar
        # create new colormap
        z = np.exp(1j * np.linspace(-np.pi, np.pi, 512))
        rgb_vals = domain_coloring(z)
        rgba_vals = np.pad(rgb_vals, ((0, 0), (0, 1)), constant_values=1.0)
        newcmp = colors.ListedColormap(rgba_vals)
        #
        norm = colors.Normalize(vmin=-np.pi, vmax=np.pi)
        cb1 = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=newcmp))
        cb1.set_label("arg", rotation=0, ha="center", va="top")
        cb1.ax.yaxis.set_label_coords(0.5, -0.03)
        cb1.set_ticks([-np.pi, -np.pi / 2, 0, +np.pi / 2, np.pi])
        cb1.set_ticklabels(["-π", "-π/2", "0", "π/2", "π"])

plot_colors(atomic_transition_potentials[1].measure().array,add_colorbars=True)
from abtem.visualize.utils import domain_coloring
def plot_colors(
                array,
                # extent,
                add_colorbars: bool = True,
                num_abs = 5
                ):
    from cplot._colors import get_srgb1,scale01
    from matplotlib import colors,cm
    
    plt.imshow(
        domain_coloring(array),
        # extent=extent,
        interpolation="nearest",
        origin="lower",
        aspect="equal",
    )

    if add_colorbars:
        # abs colorbar
        vmin = np.min(np.abs(array))
        vmax = np.max(np.abs(array))
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
        cb0 = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cm.gray))
        cb0.set_label("abs", rotation=0, ha="center", va="top")
        cb0.ax.yaxis.set_label_coords(0.5, -0.03)
        # abs_sampling = np.linspace(vmin,vmax,num=num_abs)
        # str_abs_sampling = ["{:.1E}".format(n) for n in abs_sampling]
        # cb0.set_ticks(np.linspace(0,1,num=num_abs))
        # cb0.set_ticklabels(str_abs_sampling)

        # arg colorbar
        # # create new colormap
        # z = np.exp(1j * np.linspace(-np.pi, np.pi, 512))
        # rgb_vals = domain_coloring(z)
        # rgba_vals = np.pad(rgb_vals, ((0, 0), (0, 1)), constant_values=1.0)
        # newcmp = colors.ListedColormap(rgba_vals)
        # #
        norm = colors.Normalize(vmin=-np.pi, vmax=np.pi)
        cb1 = plt.colorbar(cm.ScalarMappable(norm=norm))
        cb1.set_label("arg", rotation=0, ha="center", va="top")
        cb1.ax.yaxis.set_label_coords(0.5, -0.03)
        cb1.set_ticks([-np.pi, -np.pi / 2, 0, +np.pi / 2, np.pi])
        cb1.set_ticklabels(["-π", "-π/2", "0", "π/2", "π"])

plot_colors(atomic_transition_potentials[1].measure().array,add_colorbars=True)
from abtem.visualize.utils import domain_coloring
def plot_colors(
                array,
                # extent,
                add_colorbars: bool = True,
                num_abs = 5
                ):
    from cplot._colors import get_srgb1,scale01
    from matplotlib import colors,cm
    
    plt.imshow(
        domain_coloring(array),
        # extent=extent,
        interpolation="nearest",
        origin="lower",
        aspect="equal",
    )

    if add_colorbars:
        # abs colorbar
        vmin = np.min(np.abs(array))
        vmax = np.max(np.abs(array))
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
        cb0 = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cm.gray))
        cb0.set_label("abs", rotation=0, ha="center", va="top")
        cb0.ax.yaxis.set_label_coords(0.5, -0.03)
        # abs_sampling = np.linspace(vmin,vmax,num=num_abs)
        # str_abs_sampling = ["{:.1E}".format(n) for n in abs_sampling]
        # cb0.set_ticks(np.linspace(0,1,num=num_abs))
        # cb0.set_ticklabels(str_abs_sampling)

        # arg colorbar
        # # create new colormap
        z = np.exp(1j * np.linspace(-np.pi, np.pi, 1024))
        rgb_vals = domain_coloring(z)
        rgba_vals = np.pad(rgb_vals, ((0, 0), (0, 1)), constant_values=1.0)
        newcmp = colors.ListedColormap(rgba_vals)
        # #
        norm = colors.Normalize(vmin=-np.pi, vmax=np.pi)
        cb1 = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=newcmp))
        cb1.set_label("arg", rotation=0, ha="center", va="top")
        cb1.ax.yaxis.set_label_coords(0.5, -0.03)
        cb1.set_ticks([-np.pi, -np.pi / 2, 0, +np.pi / 2, np.pi])
        cb1.set_ticklabels(["-π", "-π/2", "0", "π/2", "π"])

plot_colors(atomic_transition_potentials[1].measure().array,add_colorbars=True)
from abtem.visualize.utils import domain_coloring
def plot_colors(
                array,
                # extent,
                add_colorbars: bool = True,
                num_abs = 5
                ):
    from cplot._colors import get_srgb1,scale01
    from matplotlib import colors,cm
    
    plt.imshow(
        domain_coloring(array),
        # extent=extent,
        interpolation="nearest",
        origin="lower",
        aspect="equal",
    )

    if add_colorbars:
        # abs colorbar
        vmin = np.min(np.abs(array))
        vmax = np.max(np.abs(array))
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
        cb0 = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cm.gray))
        cb0.set_label("abs", rotation=0, ha="center", va="top")
        cb0.ax.yaxis.set_label_coords(0.5, -0.03)
        # abs_sampling = np.linspace(vmin,vmax,num=num_abs)
        # str_abs_sampling = ["{:.1E}".format(n) for n in abs_sampling]
        # cb0.set_ticks(np.linspace(0,1,num=num_abs))
        # cb0.set_ticklabels(str_abs_sampling)

        # arg colorbar
        # # create new colormap
        # z = np.exp(1j * np.linspace(-np.pi, np.pi, 128))
        # rgb_vals = domain_coloring(z)
        # rgba_vals = np.pad(rgb_vals, ((0, 0), (0, 1)), constant_values=1.0)
        # newcmp = colors.ListedColormap(rgba_vals)
        # #
        norm = colors.Normalize(vmin=-np.pi, vmax=np.pi)
        cb1 = plt.colorbar(cm.ScalarMappable(norm=norm, cmap='cam16'))
        cb1.set_label("arg", rotation=0, ha="center", va="top")
        cb1.ax.yaxis.set_label_coords(0.5, -0.03)
        cb1.set_ticks([-np.pi, -np.pi / 2, 0, +np.pi / 2, np.pi])
        cb1.set_ticklabels(["-π", "-π/2", "0", "π/2", "π"])

plot_colors(atomic_transition_potentials[1].measure().array,add_colorbars=True)
from abtem.visualize.utils import domain_coloring
def plot_colors(
                array,
                # extent,
                add_colorbars: bool = True,
                num_abs = 5
                ):
    from cplot._colors import get_srgb1,scale01
    from matplotlib import colors,cm
    
    plt.imshow(
        domain_coloring(array),
        # extent=extent,
        interpolation="nearest",
        origin="lower",
        aspect="equal",
    )

    if add_colorbars:
        # abs colorbar
        vmin = np.min(np.abs(array))
        vmax = np.max(np.abs(array))
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
        cb0 = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cm.gray))
        cb0.set_label("abs", rotation=0, ha="center", va="top")
        cb0.ax.yaxis.set_label_coords(0.5, -0.03)
        # abs_sampling = np.linspace(vmin,vmax,num=num_abs)
        # str_abs_sampling = ["{:.1E}".format(n) for n in abs_sampling]
        # cb0.set_ticks(np.linspace(0,1,num=num_abs))
        # cb0.set_ticklabels(str_abs_sampling)

        # arg colorbar
        # create new colormap
        z = np.exp(1j * np.linspace(-np.pi, np.pi, 128))
        rgb_vals = domain_coloring(z)
        rgba_vals = np.pad(rgb_vals, ((0, 0), (0, 1)), constant_values=1.0)
        newcmp = colors.ListedColormap(rgba_vals)
        # #
        norm = colors.Normalize(vmin=-np.pi, vmax=np.pi)
        cb1 = plt.colorbar(cm.ScalarMappable(norm=norm, cmap='cam16'))
        cb1.set_label("arg", rotation=0, ha="center", va="top")
        cb1.ax.yaxis.set_label_coords(0.5, -0.03)
        cb1.set_ticks([-np.pi, -np.pi / 2, 0, +np.pi / 2, np.pi])
        cb1.set_ticklabels(["-π", "-π/2", "0", "π/2", "π"])

plot_colors(atomic_transition_potentials[1].measure().array,add_colorbars=True)
from abtem.visualize.utils import domain_coloring
def plot_colors(
                array,
                # extent,
                add_colorbars: bool = True,
                num_abs = 5
                ):
    from cplot._colors import get_srgb1,scale01
    from matplotlib import colors,cm
    
    plt.imshow(
        domain_coloring(array),
        # extent=extent,
        interpolation="nearest",
        origin="lower",
        aspect="equal",
    )

    if add_colorbars:
        # abs colorbar
        vmin = np.min(np.abs(array))
        vmax = np.max(np.abs(array))
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
        cb0 = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cm.gray))
        cb0.set_label("abs", rotation=0, ha="center", va="top")
        cb0.ax.yaxis.set_label_coords(0.5, -0.03)
        # abs_sampling = np.linspace(vmin,vmax,num=num_abs)
        # str_abs_sampling = ["{:.1E}".format(n) for n in abs_sampling]
        # cb0.set_ticks(np.linspace(0,1,num=num_abs))
        # cb0.set_ticklabels(str_abs_sampling)

        # arg colorbar
        # create new colormap
        z = np.exp(1j * np.linspace(-np.pi, np.pi, 128))
        rgb_vals = domain_coloring(z)
        rgba_vals = np.pad(rgb_vals, ((0, 0), (0, 1)), constant_values=1.0)
        newcmp = colors.ListedColormap(rgba_vals)
        # #
        norm = colors.Normalize(vmin=-np.pi, vmax=np.pi)
        cb1 = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=newcmp))
        cb1.set_label("arg", rotation=0, ha="center", va="top")
        cb1.ax.yaxis.set_label_coords(0.5, -0.03)
        cb1.set_ticks([-np.pi, -np.pi / 2, 0, +np.pi / 2, np.pi])
        cb1.set_ticklabels(["-π", "-π/2", "0", "π/2", "π"])

plot_colors(atomic_transition_potentials[1].measure().array,add_colorbars=True)
atomic_transition_potentials[1].show(cbar=True)
