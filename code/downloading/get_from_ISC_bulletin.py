import urllib.request
import html2text
import obspy
import pickle

save_path = '/home/earthquakes1/homes/Rebecca/phd/stf/data/isc/'


def build_link(start_year=2018,
               start_month=1,
               start_day=1,
               end_year=2021,
               end_month=3,
               end_day=1,
               file_type="ISF2"):
    link = ("http://www.isc.ac.uk/cgi-bin/web-db-run?"
            f"&request=COMPREHENSIVE&out_format={file_type}"
            "&bot_lat=&top_lat=&left_lon=&right_lon="
            "&ctr_lat=&ctr_lon="
            "&radius=&max_dist_units=deg"
            "&searchshape=GLOBAL"
            "&srn=&grn="
            f"&start_year={start_year}&start_month={start_month}"
            f"&start_day={start_day}&start_time=00%3A00%3A00"
            f"&end_year={end_year}&end_month={end_month}"
            f"&end_day={end_day}&end_time=00%3A00%3A00"
            "&min_dep=&max_dep="
            "&min_mag=5.5&max_mag="
            "&req_mag_type=&req_mag_agcy=GCMT"
            "&min_def=&max_def="
            "&include_magnitudes=on"
            "&include_links=on"
            "&include_headers=on"
            "&include_comments=on")
    return link


def read_bulletin_from_web_and_save():
    link = build_link(file_type="ISF2")
    f = urllib.request.urlretrieve(link, 'test.txt')
    j = open('test.txt')
    k = j.read()
    rendered_content = html2text.html2text(k)
    j.close()
    with open('rendered_content.txt', 'w') as f:
        f.write(rendered_content)
    cat_link = build_link(file_type="QUAKEML")
    urllib.request.urlretrieve(cat_link, 'quakeml.xml')


def read_from_file():
    quakeml = obspy.read_events('quakeml.xml')
    with open("rendered_content.txt", "r") as f:
        r = f.read()
    return r, quakeml


r, quakeml = read_from_file()
sp = r.split('[Event ')
sp[-1] = sp[-1].split('STOP')[0]
count = 0
for i in range(1, len(sp)):
    stf_exists = False
    cont = True
    eq_id = sp[i].split(']')[0]
    # if len(eq_id)==9:
    #    try:
    #        int(eq_id)
    #        # ---> earthquake id
    #    except:
    #        cont = False
    #        continue
    if cont is True:
        print(eq_id, "+++")
        split_at_new_lines = sp[i].split('\n')
        for s in range(0, len(split_at_new_lines)):
            if '#NORMALISED_STF' in split_at_new_lines[s]:
                stf_exists = True
                stf = split_at_new_lines[s + 1:s + 33]
                stf_list = []
                for line in stf:
                    for item in line.split(' '):
                        if item not in ['', '#', '(#', ')']:
                            stf_list.append(float(item))
            if "#STF" in split_at_new_lines[s]:
                values = split_at_new_lines[s + 1]
                values = values.split('#')[1].split()
                print(values)
                norm_dict = {'N_samp': int(values[0]),
                             'FS': int(values[1]),
                             'mo_norm': float(values[2]),
                             'STF_units': values[3],
                             'NST': int(values[4]),
                             'NWV': int(values[5]),
                             'Author': values[6]}
        if stf_exists is True:
            print('stf exists')
            for event in quakeml:
                # print(event.resource_id.id[-9:])
                if event.resource_id.id[-9:] == eq_id:
                    print('same')

                    filename = f'{save_path}/{eq_id}.xml'
                    event.write(filename, format="QUAKEML")
                    with open(f'{save_path}{eq_id}.txt', 'wb') as f:
                        pickle.dump(stf_list, f)
                    with open(f'{save_path}{eq_id}_norm_info.txt', 'wb') as f:
                        pickle.dump(norm_dict, f)
                    print('saved')
                    count += 1
print(count)
