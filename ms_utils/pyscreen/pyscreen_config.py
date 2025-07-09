from dataclasses import dataclass

from ms_utils.formula_annotation.isotopic_pattern import isotopic_pattern_config
from ms_utils.pyscreen.spectral_search import search_config
from ms_utils.interfaces.msdial import blank_config
from ms_utils.pyscreen.epa import suspect_list_config



adducts_pos = {
    "[M+H]+":1.007325,
    "[M+NH4]+": 18.034,
    "[M+Na]+":22.989
}

adducts_neg = {
    "[M-H]-":-1.007325
}


@dataclass
class pyscreen_config:
    search:search_config=None
    isotopic_pattern:isotopic_pattern_config=None
    blank:blank_config=None
    suspect_list:suspect_list_config=None
    def __post_init__(self):
        if(isinstance(self.search,dict)):
            self.search = search_config(**self.search)
        if self.search is None:
            raise Exception('search configuration is not set! you need to give me at least polarity you know')
        
        if(isinstance(self.isotopic_pattern,dict)):
            self.isotopic_pattern = isotopic_pattern_config(**self.isotopic_pattern)
        if self.isotopic_pattern is None:
            pass #TODO: hnadle this better or add a check laer for nullity of isotopic pattern
        
        if(isinstance(self.blank,dict)):
            self.blank = blank_config(**self.blank)
        if self.blank is None:
            self.blank = blank_config()
        
        if(isinstance(self.suspect_list,dict)):
            self.suspect_list = suspect_list_config(**self.suspect_list)
        if self.suspect_list is None:
            self.suspect_list = suspect_list_config()

    def to_dict(self):
        return {
            'search':self.search.to_dict(),
            'isotopic_pattern':self.isotopic_pattern.to_dict(),
            'blank':self.blank.to_dict(),
            'suspect_list':self.suspect_list.to_dict()
        }
    
    @classmethod
    def from_dict(cls,config:dict):
        if isinstance(config, cls): # because mistakes happen
            return config
        # now make sure that the config is a dictionary
        if not isinstance(config, dict):
            raise TypeError("config must be a dictionary or a pyscreen_config object")
        #now generate each subconfig from the dict
        # we fisrt check if its alredy the right type, anmd if not and its a, dict we use the from_dict method of the subconfig class
        if 'search' in config.keys():
            if isinstance(config['search'], search_config):
                search_config_instance = config['search']
            elif isinstance(config['search'], dict):
                search_config_instance = search_config.from_dict(config['search'])
            else:
                raise TypeError(f"search must be a search_config object or a dictionary, but got {type(config['search'])}")
        else:
            search_config_instance = None

        if 'isotopic_pattern' in config.keys():
            if isinstance(config['isotopic_pattern'], isotopic_pattern_config):
                isotopic_pattern_instance = config['isotopic_pattern']
            elif isinstance(config['isotopic_pattern'], dict):
                isotopic_pattern_instance = isotopic_pattern_config.from_dict(config['isotopic_pattern'])
            else:
                isotopic_pattern_instance = None
        else:
            isotopic_pattern_instance = None

        if 'blank' in config.keys():
            if isinstance(config['blank'], blank_config):
                blank_instance = config['blank']
            elif isinstance(config['blank'], dict):
                blank_instance = blank_config.from_dict(config['blank'])
            else:
                blank_instance = None
        else:
            blank_instance = None

        if 'suspect_list' in config.keys():
            if isinstance(config['suspect_list'], suspect_list_config):
                suspect_list_instance = config['suspect_list']
            elif isinstance(config['suspect_list'], dict):
                suspect_list_instance = suspect_list_config.from_dict(config['suspect_list'])
            else:
                suspect_list_instance = None
        else:
            suspect_list_instance = None

        return cls(
            search=search_config_instance,
            isotopic_pattern=isotopic_pattern_instance,
            blank=blank_instance,
            suspect_list=suspect_list_instance
        )

        

if __name__ == "__main__":

    # print(blank_config(**{
    #         'ms1_mass_tolerance':3e-6,
    #         'dRT_min':0.1,
    #         'ratio':5, 
    #         'use_ms2':False,
    #         'dRT_min_with_ms2':0.5, 
    #         'ms2_fit':0.85
    #     }))

    config = {
        'search':
        {
            'polarity':'positive',
            'NIST_db_path': '/home/analytit_admin/Data/NIST_hr_msms/NIST_hr_msms.parquet',
        },
        'isotopic_pattern':
        {
            'mass_tolerance' : 3*1e-6, 
            'max_intensity_ratio' : 1.7,
            'minimum_intensity' : 1e5,
            'ms1_resolution':0.7e5,
        },
        'blank':
        {
            'ms1_mass_tolerance':3e-6,
            'dRT_min':0.1,
            'ratio':5, 
            'use_ms2':False,
            'dRT_min_with_ms2':0.5, 
            'ms2_fit':0.85
        },
        'suspect_list':
        {
            "epa_db_path": "/home/analytit_admin/Data/EPA/suspect_list.parquet",
        }
    }
    config = pyscreen_config.from_dict(config)
    print(config)