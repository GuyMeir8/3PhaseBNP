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
                        'Liquid' : lambda T : 18
                    },
                    'Vacuum' : lambda T : 1.675 - 0.47e-3*T
                },
                'Liquid' : { # Liquid Ag
                    'Ag' : {
                        'FCC' : lambda T : 0.15*(1.675 - 0.47e-3*T),
                        'Liquid' : lambda T : 0.025
                    },
                    'Cu' : {
                        'FCC': lambda T: 18,
                        'Liquid': lambda T: 0.025
                    },
                    'Vacuum' : lambda T : 1e-3*(1207-0.228*T)
                }

            },
            'Cu' : {
                'FCC': { # FCC Cu
                    'Ag': {
                        'FCC': lambda T: (2.158512 - 0.4e-3*T +1.675 - 0.47e-3*T )/6,
                        'Liquid': lambda T: 18
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
                        'FCC': lambda T: 18,
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
