from .dtu import MVSDatasetDTU
from .llff import LLFFDataset
from .neural3Dvideo import Neural3DvideoDataset

dataset_dict = {'dtu': MVSDatasetDTU,
                'llff': LLFFDataset
                'neural3Dvideo': Neural3DvideoDataset
                }
