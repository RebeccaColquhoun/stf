'''
event class
'''

class Event():

    '''
    Class for earthquake calculations
    Args required:
        * name      : Instance Name (eq_id)
    '''

    # -------------------------------------------------------------------------------
    # Initialize class #
    def __init__(self, name):

        '''
        Initialize main variables of the class
        * name -- instance name -- strings
        '''
        isinstance(name, str), 'name argument must be a string'
        self.name = name

    def get_mag_df(self, sp):
        import pandas as pd
        mag_table = ((sp.split('Magnitude')[1]).replace('\n',' ').partition('OrigID')[-1].partition('[')[0])
        l_mags = mag_table[-1].lstrip().split(' ')
        single_mag_list = []
        start = 0
        for l in range(len(l_mags)):
            if len(l_mags[l])==9:
                single_mag_list.append(l_mags[start:l+1])
                start = l+1
        df = pd.DataFrame(columns = ['mag_type', 'mag','err','Nsta','author','orig_id'])
        for mag_line in single_mag_list:
            posn = 0
            mag_type, mag, err, Nsta, author, orig_id = '', None, None, None, '',''
            if mag_line[posn][0].lower()=='m':
                mag_type = mag_line[posn]
                posn += 1
            else:
                mag_type = ''
            if '.' in mag_line[posn]:
                mag = float(mag_line[posn])
                posn += 1
            if '.' in mag_line[posn]:
                print(posn)
                err = float(mag_line[posn])
                print(err)
                posn += 1
            try:
                int(mag_line[posn])
                Nsta = int(mag_line[posn])
                posn += 1
            except Exception:
                pass
            if mag_line[posn].isalpha():
                author = mag_line[posn]
                posn += 1
            if len(mag_line[posn])==9:
                orig_id = mag_line[posn]
                posn += 1
            print(mag_line)
            print(mag_type, mag, err, Nsta, author, orig_id)
            row = {'mag_type':mag_type, 'mag':mag,'err':err,'Nsta':Nsta,'author':author,'orig_id':orig_id}
            df = pd.concat([df, pd.DataFrame(row, index=[0])])
        self.mag = df


                        
