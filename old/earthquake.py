'''
earthquake class
'''

class Earthquake():

    '''
    Class for earthquake calculations
    Args required:
        * name      : Instance Name (corresponding to foldername, e.g. 20120101_102342.a)
        * catalog_object  : Obspy catalog containing 1 event
    '''

    # -------------------------------------------------------------------------------
    # Initialize class #
    def __init__(self, name):

        '''
        Initialize main variables of the class
        * name -- instance name -- strings
        '''
        isinstance(name, str), 'name argument must be a string'
        self.name(name)
