import os


class VehicleID():

    dataset_name = 'VehicleID_V1.0'                            
    txt_files = {
        'train': 'train_list.txt',
        'test800': 'test_list_800.txt',
        'test1600': 'test_list_1600.txt',
        'test2400': 'test_list_2400.txt',
        'test3200': 'test_list_3200.txt',
        'test6000': 'test_list_6000.txt',
        'test13164': 'test_list_13164.txt',
    }

    def __init__(self, root='/home/zhuxianpeng/dataset', verbose=True, **kwargs):
        self.image_folder = os.path.join(root, self.dataset_name, 'image')
        self.txt_folder = os.path.join(root, self.dataset_name, 'train_test_split')

        self.train, self.num_train_pids = self.get_data_label(self.image_folder, self.txt_files['train'], relabel=True)
        
        self.test800, self.num_test800_ids = self.get_data_label(self.image_folder, self.txt_files['test800'])
        """
        self.test1600, self.num_test1600_ids = self.get_data_label(self.image_folder, self.txt_files['test1600'])
        self.test2400, self.num_test2400_ids = self.get_data_label(self.image_folder, self.txt_files['test2400']) 
        """
        # default query and gallery is splited from test800
        self.query, self.gallery, self.num_query_images, self.num_gallery_images, \
            self.num_query_pids, self.num_gallery_pids = self.split_query_gallery(self.test800)
        """
        self.test1600_query, self.test1600_gallery, self.num_test1600_query_images, self.num_test1600_gallery_images, \
            self.num_test1600_query_pids, self.num_test1600_gallery_pids = self.split_query_gallery(self.test1600)
        self.test2400_query, self.test2400_gallery, self.num_test2400_query_images, self.num_test2400_gallery_images, \
            self.num_test2400_query_pids, self.num_test2400_gallery_pids = self.split_query_gallery(self.test2400) 
        """
        if verbose:
            print("===> {} loaded".format(self.dataset_name))
            print('----------------------------------')
            print('subset        | # images  | # ids ')
            print('----------------------------------')
            print('train         | {:6d} | {:8d} '.format(len(self.train), self.num_train_pids))
            print('t800 query    | {:6d} | {:8d} '.format(self.num_query_images, self.num_query_pids))
            print('t800_gallery  | {:6d} | {:8d} '.format(self.num_gallery_images, self.num_gallery_pids))
            #print('t1600_query   | {:6d} | {:8d} '.format(self.num_test1600_query_images, self.num_test1600_query_pids)) 
            #print('t1600_gallery | {:6d} | {:8d} '.format(self.num_test1600_gallery_images, self.num_test1600_gallery_pids))
            #print('t2400_query   | {:6d} | {:8d} '.format(self.num_test2400_query_images, self.num_test2400_query_pids))
            #print('t2400_gallery | {:6d} | {:8d} '.format(self.num_test2400_gallery_images, self.num_test2400_gallery_pids)) 
            print('----------------------------------')

    def split_query_gallery(self, data):
        # the first image for each id is added into gallery list, others is in query list
        query_list = list()
        gallery_list = list()
        query_ids = list()
        gallery_ids = list()
        for image, vid, cam in data:
            if vid not in gallery_ids:
                gallery_ids.append(vid)
                gallery_list.append((image, vid, cam))
            else:
                if vid not in query_ids:
                    query_ids.append(vid)
                query_list.append((image, vid, cam))
        return query_list, gallery_list, len(query_list), len(gallery_list), len(query_ids), len(gallery_ids)



    def get_data_label(self, image_folder, txt_file, relabel=False):

        txt_path = os.path.join(self.txt_folder, txt_file)
        data_list = list()
        id_list = list()
        cam_id = 0 # the camera id is unique for every image to keep test process running correctly
        minlabel=0
        with open(txt_path, 'r') as f:
            line = 'start'
            while line:
                line = f.readline()
                line = line[:-1]
                if len(line) == 0:
                    break
                name, vid = line.split(' ')
                if vid not in id_list:
                    id_list.append(vid)
                image_path = name+'.jpg'
                if relabel:
                    vid = id_list.index(vid)
                if int(vid)<minlabel:
                    minlabel=int(vid)
                data_list.append((image_path, int(vid), cam_id))
                cam_id += 1
       
        return data_list, len(id_list)



if __name__ == "__main__":
    dataset = VehicleID(root='/DATA/sunxia/', verbose=True)
    train = dataset.train
    #query = dataset.query
    #gallery = dataset.gallery
    #print(train)
    #l = [train]#, query, gallery]
    """
    for data in l:
        print('==============================')
        for i in range(10):
            print(data[i])
            print(os.path.exists(data[i][0]))
    """