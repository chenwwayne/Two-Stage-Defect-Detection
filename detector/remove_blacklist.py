import os

def remove_blacklist(file_list, blacklist_txt=None):
    if blacklist_txt is None:
        blacklist_txt = '/root/T2/dataset/1350/blacklist.dat'
    if blacklist_txt == '':
        return file_list
    blacklist = {}
    with open(blacklist_txt) as f:
        for line in f.readlines():
            line = line.strip().split()
            if len(line) == 0:
                continue
            key = line[0]
            if len(line) == 1:
                blacklist[key] = None
            elif len(line) == 2:
                blacklist[key] = line[1]
            else:
                raise ''
    removed_list = []
    print('\n### removed_list')
    for s in file_list:
        if isinstance(s, str):
            t = s.strip().split()
        else:
             t = s
        key = os.path.join(t[1], os.path.basename(t[0]))
        if key in blacklist.keys():
            value = blacklist[key]
            if value is None:
                # print('{}: {} -> {}'.format(t[0], t[1], 'None'))
                continue
            else:
                old_value = s[1]
                s[1] = value
                # print('{}: {} -> {}'.format(t[0], old_value, t[1]))
        removed_list.append(s)
    print('remove file: ', len(file_list) - len(removed_list))
    print('###\n')
    return removed_list
    
    