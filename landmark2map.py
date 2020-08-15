def cords_to_map(cords, device='cuda:0', sigma=6):
    batchsize = cords.size(0)
    result = torch.zeros((batchSize,106,256,256), device=device)
    for idx,cord in enumerate(cords):
        if idx%2==0:
            x = cord
        else:
            y = cord






    for n,k in enumerate(KEYS):
        for j,cord in enumerate(cords):
            for i, point in enumerate(cord):
                if point[0] == MISSING_VALUE or point[1] == MISSING_VALUE:
                    continue
                if i not in IDS[k]:
                    continue
                if affine_matrix is not None:
                    point_ =np.dot(affine_matrix[j], np.matrix([point[1], point[0], 1]).reshape(3,1))
                    point_0 = int(point_[1])
                    point_1 = int(point_[0])
                else:
                    point_0 = int(point[0])
                    point_1 = int(point[1])
                xx, yy = torch.meshgrid(torch.arange(256.,device=device), torch.arange(256.,device=device))
                result[n*opt.batchSize+j,i,...] = torch.exp(-((xx - point_0) ** 2 + (yy - point_1) ** 2) / (2 * sigma ** 2))
    return result