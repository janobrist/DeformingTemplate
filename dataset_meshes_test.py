from torch.utils.data import Dataset
import torch
import os
import trimesh

class Dataset_mesh(Dataset):
    def __init__(self, data_root):
        self.data_root = data_root
        self.samples = []
        i = 0

        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")
            print("WARNING: CPU only, this will be slow!")

        for meshFile in os.listdir(data_root):
            meshPath = os.path.join(data_root, meshFile)
            mesh_obj = trimesh.load(meshPath)
            verts_obj = mesh_obj.vertices
            faces_obj = mesh_obj.faces

            faces_obj = torch.tensor(faces_obj).float()
            verts_obj = torch.tensor(verts_obj).float()
            #print('verts shape: ', verts_obj.shape)
            #print('faces shape: ', faces_obj.shape)
            faces_idx_obj = faces_obj.to(device)
            verts_obj = verts_obj.to(device)
            center_obj = verts_obj.mean(0)

            verts_obj = verts_obj - center_obj

            #x -= x.mean(0)
            #d = np.sqrt((x ** 2).sum(1))
            #x /= d.max()

            #scale_obj = max(verts_obj.abs().max(0)[0])
            #verts_obj = verts_obj/scale_obj

            scale_obj = torch.sqrt((verts_obj ** 2).sum(1)).max()
            #print('###############scale_obj: ', scale_obj)
            verts_obj = verts_obj/scale_obj


            self.samples.append({'vertices': verts_obj, 'faces': faces_idx_obj, 'name': meshFile, 'center_obj': center_obj, 'scale_obj':scale_obj})

        

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

class Dataset_mesh_objects(Dataset):
    def __init__(self, trg_root, src_root, moreFaces):
        self.trg_root = trg_root
        self.src_root = src_root
        self.paths = []
        self.moreFaces=moreFaces
        i = 0

        for meshTrgFile in os.listdir(self.trg_root):
            ##########################################################################################################################
            #target
            ##########################################################################################################################
            self.paths.append(meshTrgFile)
            
        

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        #print('sample: ', self.paths[idx])
        mesh_trg_Path = os.path.join(self.trg_root, self.paths[idx])
        mesh_trg_obj = trimesh.load(mesh_trg_Path)
        verts_trg_obj = mesh_trg_obj.vertices
        #print('verts_obj shape: ', verts_trg_obj.shape)
        num_trg_points = verts_trg_obj.shape[0]
        faces_trg_obj = mesh_trg_obj.faces
        num_trg_faces = faces_trg_obj.shape[0]

        faces_trg_obj = torch.tensor(faces_trg_obj).float()
        verts_trg_obj = torch.tensor(verts_trg_obj).float()
        #print('verts shape: ', verts_obj.shape)
        #print('faces shape: ', faces_obj.shape)
        faces_trg_idx_obj = faces_trg_obj#.to(self.device)
        verts_trg_obj = verts_trg_obj#.to(self.device)
        center_trg_obj = verts_trg_obj.mean(0)

        verts_trg_obj = verts_trg_obj - center_trg_obj

        #x -= x.mean(0)
        #d = np.sqrt((x ** 2).sum(1))
        #x /= d.max()

        #scale_obj = max(verts_obj.abs().max(0)[0])
        #verts_obj = verts_obj/scale_obj

        scale_trg_obj = torch.sqrt((verts_trg_obj ** 2).sum(1)).max()
        #print('###############scale_obj: ', scale_obj)
        verts_trg_obj = verts_trg_obj/scale_trg_obj
        ##########################################################################################################################
        #src
        ##########################################################################################################################
        nameSrc = self.paths[idx][:-9]+'.off'
        if(self.moreFaces):
            nameSrc='car2more.off'#'car.off'
        else:
            nameSrc='car.off'
        print('nameSrc: ', nameSrc)
        mesh_src_Path = os.path.join(self.src_root, nameSrc)
        print('the path: ', mesh_src_Path)
        mesh_src_obj = trimesh.load(mesh_src_Path)
        verts_src_obj = mesh_src_obj.vertices
        print('verts_obj shape: ', verts_src_obj.shape)
        print('verts: ', verts_src_obj)
        
        num_src_points = verts_src_obj.shape[0]
        faces_src_obj = mesh_src_obj.faces
        print('faces_obj shape: ', faces_src_obj.shape)
        num_src_faces = faces_src_obj.shape[0]

        faces_src_obj = torch.tensor(faces_src_obj).float()
        verts_src_obj = torch.tensor(verts_src_obj).float()
        print(torch.unique(torch.isinf(verts_src_obj)))
        print(torch.unique(torch.isnan(verts_src_obj)))
        print(torch.unique(torch.isinf(faces_src_obj)))
        print(torch.unique(torch.isnan(faces_src_obj)))
        #print('verts shape: ', verts_obj.shape)
        #print('faces shape: ', faces_obj.shape)
        faces_src_idx_obj = faces_src_obj#.to(self.device)
        verts_src_obj = verts_src_obj#.to(self.device)
        center_src_obj = verts_src_obj.mean(0)

        verts_src_obj = verts_src_obj - center_src_obj
        scale_src_obj = torch.sqrt((verts_src_obj ** 2).sum(1)).max()
        #print('###############scale_obj: ', scale_obj)
        verts_src_obj = verts_src_obj/scale_src_obj

        #print('heeeeeeeeeeeeeeeerrrrrrrrrrrrreeeeeeeeeee')
        return {'vertices_src': verts_src_obj, 'faces_src': faces_src_idx_obj, 'vertices_trg': verts_trg_obj, 'faces_trg': faces_trg_idx_obj, 'name': self.paths[idx], 'center_obj': center_trg_obj, 'scale_obj':scale_trg_obj, 'num_points':num_trg_points, 'num_faces':num_trg_faces, 'scale_src':scale_src_obj, 'center_src':center_src_obj}

def collate_fn(data):
    """
       data: is a list of tuples with (example, label, length)
             where 'example' is a tensor of arbitrary shape
             and label/length are scalars
    """
    #print('data[0]: ', data[0])
    #device='cuda:0'
    maxPoints = 50000
    maxFaces = 70000
    #_, labels, lengths = zip(*data)
    #max_len = max(lengths)
    #n_ftrs = data[0][0].size(1)
    features_verts_trg = torch.zeros((len(data), maxPoints, 3))
    features_faces_trg = torch.zeros((len(data), maxFaces, 3))
    features_verts_src = torch.zeros((len(data), maxPoints, 3))
    features_faces_src = torch.zeros((len(data), maxFaces, 3))
    #labels = torch.tensor(labels)
    #lengths = torch.tensor(lengths)

    for i in range(len(data)):
        verts_trg = data[i]['vertices_trg']
        faces_trg = data[i]['faces_trg']
        num_points = data[i]['num_points']
        num_faces = data[i]['num_faces']
        #print('faces trg shape: ', faces_trg.shape)
        #print('vertices trg shape: ', verts_trg.shape)
        print('num_points: ', num_points)
        print('num_faces: ', num_faces)
        features_verts_trg[i] = torch.cat((verts_trg, torch.zeros((maxPoints - num_points, 3))), dim=0)
        print('trg shape verts after cat', features_verts_trg[i].shape)
        features_faces_trg[i] = torch.cat((faces_trg, torch.zeros((maxFaces - num_faces, 3))), dim=0)
        print('trg shape faces after cat', features_faces_trg[i].shape)
        
        verts_src = data[i]['vertices_src']
        faces_src = data[i]['faces_src']
        #print('faces src shape: ', faces_src.shape)
        num_points = data[i]['vertices_src'].shape[0]#data[i]['num_points']
        num_faces = data[i]['faces_src'].shape[0]#data[i]['num_faces']
        print(num_points)
        print(num_faces)
        features_verts_src[i] = torch.cat((verts_src, torch.zeros((maxPoints - num_points, 3))), dim=0)

        features_faces_src[i] = torch.cat((faces_src, torch.zeros((maxFaces - num_faces, 3))), dim=0)

        #print('features shape: ', features_verts_src[i].shape)
        #print('faces: ', data[i]['faces_src'].shape)
    
    #print('features shape: ', features_verts_src.shape)

    verts_trg = features_verts_trg
    faces_trg = features_faces_trg

    #print('faces src 0: ', faces_src[0])
    #print('faces trg 0: ', faces_trg[0])

    verts_src = features_verts_src
    faces_src = features_faces_src

    #faces = torch.cat([el['faces'].unsqueeze(0) for el in data], dim=0)
    name = [el['name'] for el in data]
    centers = torch.cat([el['center_obj'].unsqueeze(0) for el in data], dim=0)
    scale_obj = [el['scale_obj'] for el in data]
    centers_src = torch.cat([el['center_src'].unsqueeze(0) for el in data], dim=0)
    scale_src = [el['scale_src'] for el in data]
    num_points = [el['num_points'] for el in data]
    num_faces = [el['num_faces'] for el in data]
    
    

    return {'vertices_trg': verts_trg, 'faces_trg': faces_trg, 'vertices_src': verts_src, 'faces_src': faces_src, 'name': name, 'center_obj':centers, 'scale_obj':scale_obj, 'num_points':num_points, 'num_faces':num_faces, 'center_src':centers_src, 'scale_src':scale_src}

