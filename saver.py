import shutil, os
'''
usage:
# check if the path is needed to be reused
else:
    save_dir = find_new_dir(save_dir)
# the source codes should be always saved
save_src(save_dir)
'''
def find_new_dir(base_dir):
    for i in range(999999):
        new_dir = os.path.join(base_dir, str(i))
        if not os.path.isdir(new_dir):
            break
    os.mkdir(new_dir)
    return new_dir
def save_src(save_dir, src_list):
    src_dir = os.path.join(save_dir, 'src')
    if not os.path.isdir(src_dir):
        os.mkdir(src_dir)
    src_dir = find_new_dir(src_dir)
    def safe_src_copy(src_list, dst):
        assert type(src_list) is list
        for src in src_list:
            assert os.path.isfile(src)
            dst_file_path = os.path.join(dst, os.path.basename(src))
            assert (os.path.isdir(dst) and not os.path.isfile(dst_file_path))
            shutil.copy(src, dst)
    safe_src_copy(src_list, src_dir)
