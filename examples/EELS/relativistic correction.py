def schroedinger_derivative_relativistic_correct(y, r, l, e, vr):
            # fine structure constant
            alpha = 7.2973525693e-3 
            (u, up) = y
            dvdr = (vr.derivative(n=1)(r)-vr(r)/r)/r
            # note vr is effective potential multiplied by radius:
            return np.array([up, (l * (l + 1) / r ** 2 + 2 * vr(r) / r - e - alpha**2 / 4 * ((e-2*vr(r)/r)**2+2*l/r*dvdr)) * u - alpha**2 /4 * dvdr * (up*r-u)/r])
