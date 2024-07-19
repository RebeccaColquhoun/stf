import urllib.request
import html2text
import obspy
import pickle

save_path = '/home/earthquakes1/homes/Rebecca/phd/stf/data/isc/'

download_data = True

cat_params = {'start_year': 2020,
              'start_month': 1,
              'start_day': 1,
              'end_year': 2021,
              'end_month': 1,
              'end_day': 1}


def build_link(cat_params, file_type="ISF2"):

    link = ("http://www.isc.ac.uk/cgi-bin/web-db-run?"
            f"request=COMPREHENSIVE&out_format={file_type}"
            "&bot_lat=&top_lat=&left_lon=&right_lon="
            "&ctr_lat=&ctr_lon="
            "&radius=&max_dist_units=deg"
            "&searchshape=GLOBAL"
            "&srn=&grn="
            f"&start_year={str(cat_params['start_year'])}"
            f"&start_month={str(cat_params['start_month']).zfill(2)}"
            f"&start_day={str(cat_params['start_day']).zfill(2)}"
            "&start_time=00%3A00%3A00"
            f"&end_year={str(cat_params['end_year'])}"
            f"&end_month={str(cat_params['end_month']).zfill(2)}"
            f"&end_day={str(cat_params['end_day']).zfill(2)}"
            "&end_time=00%3A00%3A00"
            "&min_dep=&max_dep="
            "&min_mag=5.5&max_mag="
            "&req_mag_type=&req_mag_agcy=GCMT"
            "&min_def=&max_def="
            "&include_magnitudes=on"
            "&include_links=on"
            "&include_headers=on"
            "&include_comments=on")
    return link


def read_bulletin_from_web_and_save(cat_params):
    link = build_link(cat_params, file_type="ISF2")
    f = urllib.request.urlretrieve(link, f'{save_path}initial_content.txt')
    j = open(f'{save_path}initial_content.txt')
    k = j.read()
    rendered_content = html2text.html2text(k)
    j.close()
    with open(f'{save_path}rendered_content.txt', 'w') as f:
        f.write(rendered_content)
    cat_link = build_link(cat_params, file_type='QuakeML')
    urllib.request.urlretrieve(cat_link, f'{save_path}quakeml.xml')
    print('bulletin downloaded')


def read_from_file():
    quakeml = obspy.read_events(f'{save_path}quakeml.xml')
    with open(f'{save_path}rendered_content.txt', 'r') as f:
        r = f.read()
    return r, quakeml


def split_bulletin_into_events(r):
    sp = r.split('[Event ')
    sp[-1] = sp[-1].split('STOP')[0]
    return sp


def look_for_stf(split_at_new_lines, s):
    if "#STF" in split_at_new_lines[s]:
        values = split_at_new_lines[s + 1]
        values = values.split('#')[1].split()
        print("#STF", values)
        norm_dict = {'N_samp': int(values[0]),
                     'FS': int(values[1]),
                     'mo_norm': float(values[2]),
                     'STF_units': values[3],
                     'NST': int(values[4]),
                     'NWV': int(values[5]),
                     'Author': values[6]}
        print(norm_dict, 'about to return')
        return norm_dict
    else:
        return None


def look_for_norm_stf(split_at_new_lines, s, stf_exists):
    stf_list = []
    if '#NORMALISED_STF' in split_at_new_lines[s]:
        print("found NORMALISED_STF")
        stf_exists = True
        stf = split_at_new_lines[s + 1:s + 33]
        #print(stf)
        for line in stf:
            for item in line.split(' '):
                #print('item', item)
                if item not in ['', '#', '(#', ')']:
                    stf_list.append(float(item))
                    #print('stflist', stf_list)
    return stf_exists, stf_list


def look_for_event_in_catalog_and_save(eq_id,
                                       count,
                                       quakeml,
                                       stf_list,
                                       norm_dict):
    for event in quakeml:
        print('eventid', event.resource_id.id[-9:])
        print(eq_id, event.resource_id.id[-9:])
        if event.resource_id.id[-9:] == eq_id:
            print('same')

            filename = f'{save_path}{eq_id}.xml'
            event.write(filename, format="QUAKEML")
            with open(f'{save_path}{eq_id}.txt', 'wb') as f:
                pickle.dump(stf_list, f)
            with open(f'{save_path}{eq_id}_norm_info.txt', 'wb') as f:
                pickle.dump(norm_dict, f)
            print('saved')
            count += 1
            break
    return count


def ISC_download_and_process_data(cat_params, download_data=True):
    print('running')

    if download_data is True:
        read_bulletin_from_web_and_save(cat_params)
        print('downloaded')

    r, quakeml = read_from_file()
    events_split = split_bulletin_into_events(r)
    print('consider each event in turn')
    count = 0


    for i in range(1, len(events_split)):

        eq_id = events_split[i].split(']')[0]
        print(eq_id, "+++")

        split_at_new_lines = events_split[i].split('\n')
        #print(split_at_new_lines)
        stf_exists = False
        norm_dict = None
        for s in range(0, len(split_at_new_lines)):
            if stf_exists is False:
                stf_exists, stf_list = look_for_norm_stf(split_at_new_lines,
                                                     s,
                                                     stf_exists)
                print(stf_exists)
            if norm_dict is None:
                norm_dict = look_for_stf(split_at_new_lines, s)
                print(norm_dict)

        if stf_exists is True:
            print('stf exists')
            count = look_for_event_in_catalog_and_save(eq_id,
                                                       count,
                                                       quakeml,
                                                       stf_list,
                                                       norm_dict)
            break

    print(count)

for i in range(2016, 2024):
    cat_params['start_year'] = i
    cat_params['end_year'] = i + 1

    ISC_download_and_process_data(cat_params, download_data=True)
