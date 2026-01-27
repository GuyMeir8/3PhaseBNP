class SurfaceEnergyValues:
    def __init__(self):
        # A a(lpha) is interacting with B b(beta)
        self.AaBb = {
            'Ag' : {
                'FCC' : { # FCC Ag
                    'Ag' : {
                        'FCC' : lambda T : 0.4,
                        'Liquid' : lambda T : 0.15
                    },
                    'Cu' : {
                        'FCC' : lambda T : 0.35,
                        'Liquid' : lambda T : 0.275
                    },
                    # 'Vacuum' : lambda T : 1.675 - 0.47e-3*T
                    'Vacuum': lambda T:2.158512 - 0.4e-3*T
                },
                'Liquid' : { # Liquid Ag
                    'Ag' : {
                        'FCC' : lambda T : 0.15,
                        'Liquid' : lambda T : 0.0
                    },
                    'Cu' : {
                        'FCC': lambda T: 0.275,
                        'Liquid': lambda T: 0.0
                    },
                    # 'Vacuum' : lambda T : 1e-3*(1207-0.228*T)
                    'Vacuum': lambda T: 1e-3*(1207-0.228*T)
                }

            },
            'Cu' : {
                'FCC': { # FCC Cu
                    'Ag': {
                        'FCC': lambda T: 0.35,
                        'Liquid': lambda T: 0.275
                    },
                    'Cu': {
                        'FCC': lambda T: 0.6,
                        'Liquid': lambda T: 0.225
                    },
                    # 'Vacuum' : lambda T : 2.158512 - 0.4e-3*T
                    'Vacuum': lambda T: 2.158512 - 0.4e-3*T
                },
                'Liquid': { # Liquid Cu
                    'Ag': {
                        'FCC': lambda T: 0.275,
                        'Liquid': lambda T: 0.0
                    },
                    'Cu': {
                        'FCC': lambda T: 0.225,
                        'Liquid': lambda T: 0
                    },
                    # 'Vacuum' : lambda T : 1e-3*(1585-0.211*T)
                    'Vacuum': lambda T: 1e-3*(1207-0.228*T)
                }
            }
        }
