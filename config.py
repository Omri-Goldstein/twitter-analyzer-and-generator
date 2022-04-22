"""General configs for all modules."""


class UsersConfig:
    def __init__(self):
        self.kernel_1 = ['ReutZeiri', 'OriKatz3', 'CiItay', 'hasolidit', 'amsterdamski2', 'assafkott', 'assafzim',
                         'itamarcaspi', 'DanieliOren', 'zeevikim']
        self.kernel_2 = ['ZRTZR', '_spez_', 'idoklein1', 'considerthefish', 'naamuli84', 'RosinJonathan',
                         'allonsigler',
                         'elad3', 't_jaulus', 'ylerman']
        self.kernel_3 = ['therealnirs', 'TalZackon', 'tomerdean', 'hitechproblems', 'shaulmert', 'rinaarts', 'barzik',
                         'amitaiz', 'EtiNoked', 'Arbel2025']
        self.kernel_4 = ['amit_segal', 'Nadav_Eyal', 'RavivDrucker', 'BarakRavid', 'BenCaspit', 'avishaigrinzaig',
                         'akivanovick', 'KalmanLiebskind', 'baruchikra', 'AyalaHasson']
        self.kernel_5 = ['Riklin10', 'YinonMagal', 'ErelSegal', 'elizipori', 'GadiTaub1', 'AvishayBenHaim', 'sharongal']
        self.kernel_6 = ['horowitz_b', 'rutihersh', 'DrorMevorach', 'feldman_gil', 'LittleMoiz', 'yuvharpaz',
                         'HillaLeon',
                         'LiorHecht', 'segal_eran', 'AArgoetti']
        self.kernel_7 = ['UdiQimron', 'DrShahar', 'mdkatrsk', 'InbarRotem', 'PECC_Israel', 'YaffaRaz', 'daridor',
                         'nur_ro1', 'guymeroz', 'LassYoram']
        self.kernel_8 = ['Ynav', 'SBogus', 'FruminJenia', 'IgalLiverant']

    def get_all_users(self):
        all_users = []
        for users_list in self.__dict__.values():
            all_users += users_list

        return all_users


class GraphConfig:
    def __init__(self):
        pass


class NetworkConfig:
    def __init__(self):
        self.batch_size = 32
        self.epochs = 5
        self.learning_rate = 3e-5
        self.max_seq_len = 128
        self.english_model = 'gpt2'
        self.small_hebrew_model = 'Norod78/hebrew-gpt_neo-small'
        self.large_hebrew_model = 'Norod78/hebrew-gpt_neo-xl'
        self.language = 'hebrew'
        self.model_size = 'small'


class GenerationConfig:
    def __init__(self):
        self.max_len = 256
        self.sample_output_num = 3
        self.seed = 42
        self.top_k = 50
        self.top_p = 0.95


class RunConfig:
    def __init__(self):
        self.override = False
        self.get_tweets = True
        self.train_model = True
        self.generate_text = True
        self.models_path = 'trained_models'
        self.tweets_path = 'tweets'
        self.build_graph = False
        self.save_tweets_per_community = False
        self.gephi_file = 'communities_from_gephi_2.csv'


class Config:
    def __init__(self):
        self.users = UsersConfig().get_all_users()
        self.graph_config = GraphConfig()
        self.network_config = NetworkConfig()
        self.generation_config = GenerationConfig()
        self.run_config = RunConfig()
