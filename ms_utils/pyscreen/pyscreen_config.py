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
        },
        'isotopic_pattern':
        {
            'mass_tolerance' : 3*1e-6, 
            'max_intensity_ratio' : 1.7,
            'minimum_intensity' : 1e5,
            'ms1_resolution':0.7e5,
        },
    }
    config = pyscreen_config(**config)
    print(config)