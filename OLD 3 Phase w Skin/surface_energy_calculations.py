from math import log, exp

class SurfaceEnergyValues:
    def __init__(self):
        # A a(lpha) is interacting with B b(beta)
        self.AaBb = {
            'Ag' : {
                'FCC' : { # FCC Ag
                    'Ag' : {
                        'FCC' : lambda T : (1.675 - 0.47e-3*T)/3,
                        'Liquid' : lambda T : 0.15*(1.675 - 0.47e-3*T)
                    },
                    'Cu' : {
                        'FCC' : lambda T : (2.158512 - 0.4e-3*T +1.675 - 0.47e-3*T )/6,
                        'Liquid' : lambda T : self._calc_AgSolid_CuLiquid(T)
                    },
                    'Vacuum' : lambda T : 1.675 - 0.47e-3*T
                },
                'Liquid' : { # Liquid Ag
                    'Ag' : {
                        'FCC' : lambda T : 0.15*(1.675 - 0.47e-3*T),
                        'Liquid' : lambda T : 0.025
                    },
                    'Cu' : {
                        'FCC': lambda T: self._calc_CuSolid_AgLiquid(T),
                        'Liquid': lambda T: 0.025
                    },
                    'Vacuum' : lambda T : 1e-3*(1207-0.228*T)
                }

            },
            'Cu' : {
                'FCC': { # FCC Cu
                    'Ag': {
                        'FCC': lambda T: (2.158512 - 0.4e-3*T +1.675 - 0.47e-3*T )/6,
                        'Liquid': lambda T: self._calc_CuSolid_AgLiquid(T),
                    },
                    'Cu': {
                        'FCC': lambda T: (2.158512 - 0.4e-3*T)/3,
                        'Liquid': lambda T: 0.15*(2.158512 - 0.4e-3*T)
                    },
                    'Ni': {
                        'FCC': lambda T: 18,
                        'Liquid': lambda T: 18
                    },
                    'Vacuum': lambda T: 2.158512 - 0.4e-3*T
                },
                'Liquid': { # Liquid Cu
                    'Ag': {
                        'FCC': lambda T: lambda T: self._calc_AgSolid_CuLiquid(T),
                        'Liquid': lambda T: 0.025
                    },
                    'Cu': {
                        'FCC': lambda T: 0.15*(2.158512 - 0.4e-3*T),
                        'Liquid': lambda T: 0.025
                    },
                    'Ni': {
                        'FCC': lambda T: 18,
                        'Liquid': lambda T: 18
                    },
                    'Vacuum' : lambda T : 1e-3*(1585-0.211*T)
                }
            },
            'Ni' : {
                'FCC': { # FCC Ni
                    'Ag': {
                        'FCC': lambda T: 0.35,
                        'Liquid': lambda T: 0.275
                    },
                    'Cu': {
                        'FCC': lambda T: 1.0,
                        'Liquid': lambda T: 0.177
                    },
                    'Ni': {
                        'FCC': lambda T: (2.940 - 3.92e-4*T)/3,
                        'Liquid': lambda T: 0.255
                    },
                    'Vacuum': lambda T: 2.940 - 3.92e-4*T
                },
                'Liquid': { # Liquid Ni
                    'Ag': {
                        'FCC': lambda T: 0.275,
                        'Liquid': lambda T: 0.025
                    },
                    'Cu': {
                        'FCC': lambda T: 0.177,
                        'Liquid': lambda T: 0.025
                    },
                    'Ni': {
                        'FCC': lambda T: 0.255,
                        'Liquid': lambda T: 0.025
                    },
                    'Vacuum' : lambda T : 2.488 - 0.393e-3*T
                }
            }
        }

    def _calc_AgSolid_CuLiquid(self, T):
        """
        Calculates the interfacial energy between Solid Ag (FCC) and Liquid Cu.
        """
        L0_FCC = 36772.58 - 11.02847*T
        L0_Liquid = 17384.37 - 4.46438 * T
        v_AgFCC = 9.9361e-6 + T * 1.1368e-9
        v_CuLiquid = 7.53e-6 + 2.49e-10*T + 1.86e-13*T**2
        f_int = 1.045
        N_A = 6.02214076e23
        w_A = f_int * (v_AgFCC ** (2/3)) * (N_A ** (1/3))
        w_B = f_int * (v_CuLiquid ** (2/3)) * (N_A ** (1/3))
        R = 8.31446261815324
        sigma_A_sl = 0.15*(1.675 - 0.47e-3*T)
        sigma_B_sl = 0.15*(2.158512 - 0.4e-3*T)
        k_int = 0.9738
        xB_int_prev = 0.45
        tol_result = 1.0
        iteration = 0
        while tol_result > 1e-6 and iteration < 1000:
            dG_A_int = k_int * (xB_int_prev**2) * (L0_FCC + L0_Liquid)/2
            dG_B_int = k_int * ((1-xB_int_prev)**2) * (L0_FCC + L0_Liquid)/2
            sigma_sl = sigma_A_sl + (R*T/w_A) * log(1-xB_int_prev) + dG_A_int / w_A
            xB_int_curr = exp((w_B/(R*T)) * (sigma_sl - sigma_B_sl - dG_B_int / w_B))
            tol_result = abs(xB_int_curr - xB_int_prev)
            iteration += 1
            xB_int_prev = xB_int_curr*0.9 + xB_int_prev*0.1
        if iteration == 1000:
            raise ValueError("Failed to converge in _calc_AgSolid_CuLiquid")
        return sigma_sl

    def _calc_CuSolid_AgLiquid(self, T):
        """
        Calculates the interfacial energy between Solid Cu (FCC) and Liquid Ag.
        """
        L0_FCC = 36772.58 - 11.02847*T
        L0_Liquid = 17384.37 - 4.46438 * T
        v_AgLiquid = 1.01961e-5+T*1.1368e-9
        v_CuFCC = 7.01e-6+2.92e-10*T+1.02e-13*T**2
        f_int = 1.045
        N_A = 6.02214076e23
        w_A = f_int * (v_AgLiquid ** (2/3)) * (N_A ** (1/3))
        w_B = f_int * (v_CuFCC ** (2/3)) * (N_A ** (1/3))
        R = 8.31446261815324
        sigma_A_sl = 0.15*(1.675 - 0.47e-3*T)
        sigma_B_sl = 0.15*(2.158512 - 0.4e-3*T)
        k_int = 0.9738
        xB_int_prev = 0.45
        tol_result = 1.0
        iteration = 0
        while tol_result > 1e-6 and iteration < 1000:
            dG_A_int = k_int * (xB_int_prev**2) * (L0_FCC + L0_Liquid)/2
            dG_B_int = k_int * ((1-xB_int_prev)**2) * (L0_FCC + L0_Liquid)/2
            sigma_sl = sigma_A_sl + (R*T/w_A) * log(1-xB_int_prev) + dG_A_int / w_A
            xB_int_curr = exp((w_B/(R*T)) * (sigma_sl - sigma_B_sl - dG_B_int / w_B))
            tol_result = abs(xB_int_curr - xB_int_prev)
            iteration += 1
            xB_int_prev = xB_int_curr*0.9 + xB_int_prev*0.1
        if iteration == 1000:
            raise ValueError("Failed to converge in _calc_AgSolid_CuLiquid")
        return sigma_sl