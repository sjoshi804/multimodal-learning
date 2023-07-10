import os
import torch
import logging
import torchvision
import pandas as pd
from PIL import Image, ImageFile
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

from utils.augment_text import _augment_text
from utils.augment_image import _augment_image

ImageFile.LOAD_TRUNCATED_IMAGES = True
    
class ImageCaptionDataset(Dataset):
    def __init__(self, path, image_key, caption_key, delimiter, processor, inmodal = False):
        logging.debug(f"Loading aligned data from {path}")

        df = pd.read_csv(path, sep = delimiter)

        self.root = os.path.dirname(path)
        self.images = df[image_key].tolist()
        self.captions = processor.process_text(df[caption_key].tolist())
        self.processor = processor
        
        self.inmodal = inmodal
        if(inmodal):
            self.augment_captions = processor.process_text([_augment_text(caption) for caption in df[caption_key].tolist()])
        
        logging.debug("Loaded data")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        item = {}
        
        if(self.inmodal):
            item["input_ids"] = self.captions["input_ids"][idx], self.augment_captions["input_ids"][idx]
            item["attention_mask"] = self.captions["attention_mask"][idx], self.augment_captions["attention_mask"][idx]
            item["pixel_values"] = self.processor.process_image(Image.open(os.path.join(self.root, self.images[idx]))), self.processor.process_image(_augment_image(os.path.join(self.root, self.images[idx])))
        else:  
            item["input_ids"] = self.captions["input_ids"][idx]
            item["attention_mask"] = self.captions["attention_mask"][idx]
            item["pixel_values"] = self.processor.process_image(Image.open(os.path.join(self.root, self.images[idx])))
            
        return item

def get_train_dataloader(options, processor):
    path = options.train_data
    if(path is None): return None

    batch_size = options.batch_size

    dataset = ImageCaptionDataset(path, image_key = options.image_key, caption_key = options.caption_key, delimiter = options.delimiter, processor = processor, inmodal = options.inmodal)
        
    sampler = DistributedSampler(dataset) if(options.distributed) else None

    dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = (sampler is None), num_workers = options.num_workers, pin_memory = True, sampler = sampler, drop_last = True)
    dataloader.num_samples = len(dataloader) * batch_size 
    dataloader.num_batches = len(dataloader)

    return dataloader

def get_validation_dataloader(options, processor):
    path = options.validation_data
    if(path is None): return

    dataset = ImageCaptionDataset(path, image_key = options.image_key, caption_key = options.caption_key, delimiter = options.delimiter, processor = processor, inmodal = options.inmodal)
    dataloader = DataLoader(dataset, batch_size = options.batch_size, shuffle = False, num_workers = options.num_workers, pin_memory = True, sampler = None, drop_last = False)
    dataloader.num_samples = len(dataset) 
    dataloader.num_batches = len(dataloader)

    return dataloader
import numpy as np


def sparse2coarse(targets):
    """Convert Pytorch CIFAR100 sparse targets to coarse targets.
    Usage:
        trainset = torchvision.datasets.CIFAR100(path)
        trainset.targets = sparse2coarse(trainset.targets)
    """
    coarse_labels = np.array([ 4,  1, 14,  8,  0,  6,  7,  7, 18,  3,  
                               3, 14,  9, 18,  7, 11,  3,  9,  7, 11,
                               6, 11,  5, 10,  7,  6, 13, 15,  3, 15,  
                               0, 11,  1, 10, 12, 14, 16,  9, 11,  5, 
                               5, 19,  8,  8, 15, 13, 14, 17, 18, 10, 
                               16, 4, 17,  4,  2,  0, 17,  4, 18, 17, 
                               10, 3,  2, 12, 12, 16, 12,  1,  9, 19,  
                               2, 10,  0,  1, 16, 12,  9, 13, 15, 13, 
                              16, 19,  2,  4,  6, 19,  5,  5,  8, 19, 
                              18,  1,  2, 15,  6,  0, 17,  8, 14, 13])
    return coarse_labels[targets]

class ImageLabelDataset(Dataset):
    def __init__(self, root, transform):
        self.root = root
        df = pd.read_csv(os.path.join(root, "labels.csv"), error_bad_lines=False)
        self.images = df["image"]
        self.labels = df["label"]
        #self.dict0 = {0: 25, 1: 25, 2: 51, 3: 51, 4: 51, 5: 25, 6: 25, 7: 6, 8: 6, 9: 6, 10: 6, 11: 6, 12: 6, 13: 6, 14: 6, 15: 6, 16: 6, 17: 6, 18: 6, 19: 6, 20: 6, 21: 6, 22: 6, 23: 6, 24: 6, 25: 50, 26: 50, 27: 50, 28: 50, 29: 50, 30: 28, 31: 28, 32: 28, 33: 60, 34: 60, 35: 60, 36: 60, 37: 60, 38: 36, 39: 36, 40: 36, 41: 36, 42: 36, 43: 36, 44: 36, 45: 36, 46: 36, 47: 36, 48: 36, 49: 16, 50: 16, 51: 19, 52: 53, 53: 53, 54: 53, 55: 53, 56: 53, 57: 53, 58: 53, 59: 53, 60: 53, 61: 53, 62: 53, 63: 53, 64: 53, 65: 25, 66: 53, 67: 53, 68: 53, 69: 59, 70: 2, 71: 2, 72: 2, 73: 2, 74: 2, 75: 2, 76: 2, 77: 2, 78: 8, 79: 8, 80: 6, 81: 6, 82: 6, 83: 6, 84: 6, 85: 6, 86: 6, 87: 6, 88: 6, 89: 6, 90: 6, 91: 6, 92: 6, 93: 6, 94: 6, 95: 6, 96: 6, 97: 6, 98: 6, 99: 6, 100: 6, 101: 61, 102: 41, 103: 41, 104: 38, 105: 38, 106: 38, 107: 25, 108: 15, 109: 15, 110: 8, 111: 8, 112: 39, 113: 39, 114: 39, 115: 39, 116: 39, 117: 39, 118: 17, 119: 17, 120: 17, 121: 17, 122: 17, 123: 17, 124: 17, 125: 17, 126: 17, 127: 6, 128: 6, 129: 6, 130: 6, 131: 6, 132: 6, 133: 6, 134: 6, 135: 6, 136: 6, 137: 6, 138: 6, 139: 6, 140: 6, 141: 6, 142: 6, 143: 6, 144: 6, 145: 6, 146: 6, 147: 37, 148: 37, 149: 37, 150: 37, 151: 20, 152: 20, 153: 20, 154: 20, 155: 20, 156: 20, 157: 20, 158: 20, 159: 20, 160: 20, 161: 20, 162: 20, 163: 20, 164: 20, 165: 20, 166: 20, 167: 20, 168: 20, 169: 20, 170: 20, 171: 20, 172: 20, 173: 20, 174: 20, 175: 20, 176: 20, 177: 20, 178: 20, 179: 20, 180: 20, 181: 20, 182: 20, 183: 20, 184: 20, 185: 20, 186: 20, 187: 20, 188: 20, 189: 20, 190: 20, 191: 20, 192: 20, 193: 20, 194: 20, 195: 20, 196: 20, 197: 20, 198: 20, 199: 20, 200: 20, 201: 20, 202: 20, 203: 20, 204: 20, 205: 20, 206: 20, 207: 20, 208: 20, 209: 20, 210: 20, 211: 20, 212: 20, 213: 20, 214: 20, 215: 20, 216: 20, 217: 20, 218: 20, 219: 20, 220: 20, 221: 20, 222: 20, 223: 20, 224: 20, 225: 20, 226: 20, 227: 20, 228: 20, 229: 20, 230: 20, 231: 20, 232: 20, 233: 20, 234: 20, 235: 20, 236: 20, 237: 20, 238: 20, 239: 20, 240: 20, 241: 20, 242: 20, 243: 20, 244: 20, 245: 20, 246: 20, 247: 20, 248: 20, 249: 20, 250: 20, 251: 20, 252: 20, 253: 20, 254: 20, 255: 20, 256: 20, 257: 20, 258: 20, 259: 20, 260: 20, 261: 20, 262: 20, 263: 20, 264: 20, 265: 20, 266: 20, 267: 20, 268: 20, 269: 66, 270: 66, 271: 66, 272: 66, 273: 66, 274: 66, 275: 20, 276: 66, 277: 66, 278: 66, 279: 66, 280: 66, 281: 11, 282: 65, 283: 11, 284: 11, 285: 11, 286: 65, 287: 65, 288: 65, 289: 65, 290: 65, 291: 65, 292: 65, 293: 65, 294: 5, 295: 5, 296: 5, 297: 52, 298: 40, 299: 40, 300: 8, 301: 8, 302: 8, 303: 8, 304: 8, 305: 8, 306: 8, 307: 8, 308: 8, 309: 8, 310: 8, 311: 8, 312: 8, 313: 8, 314: 8, 315: 8, 316: 8, 317: 8, 318: 8, 319: 8, 320: 8, 321: 10, 322: 10, 323: 10, 324: 10, 325: 10, 326: 10, 327: 21, 328: 21, 329: 21, 330: 48, 331: 48, 332: 48, 333: 49, 334: 49, 335: 49, 336: 49, 337: 49, 338: 49, 339: 46, 340: 61, 341: 33, 342: 33, 343: 33, 344: 61, 345: 61, 346: 61, 347: 61, 348: 61, 349: 61, 350: 61, 351: 61, 352: 61, 353: 61, 354: 61, 355: 61, 356: 24, 357: 24, 358: 24, 359: 24, 360: 24, 361: 24, 362: 24, 363: 3, 364: 52, 365: 47, 366: 47, 367: 47, 368: 47, 369: 47, 370: 47, 371: 47, 372: 47, 373: 47, 374: 47, 375: 47, 376: 47, 377: 47, 378: 47, 379: 47, 380: 47, 381: 47, 382: 47, 383: 40, 384: 47, 385: 61, 386: 61, 387: 5, 388: 5, 389: 25, 390: 25, 391: 51, 392: 15, 393: 25, 394: 25, 395: 25, 396: 25, 397: 25, 398: 55, 399: 12, 400: 12, 401: 34, 402: 34, 403: 7, 404: 1, 405: 1, 406: 31, 407: 63, 408: 63, 409: 18, 410: 43, 411: 12, 412: 31, 413: 64, 414: 0, 415: 9, 416: 54, 417: 57, 418: 56, 419: 42, 420: 34, 421: 31, 422: 54, 423: 31, 424: 9, 425: 9, 426: 55, 427: 13, 428: 56, 429: 4, 430: 4, 431: 31, 432: 34, 433: 32, 434: 18, 435: 31, 436: 63, 437: 9, 438: 35, 439: 32, 440: 27, 441: 27, 442: 9, 443: 12, 444: 63, 445: 12, 446: 44, 447: 56, 448: 43, 449: 9, 450: 63, 451: 0, 452: 32, 453: 31, 454: 9, 455: 42, 456: 64, 457: 0, 458: 18, 459: 12, 460: 43, 461: 12, 462: 56, 463: 56, 464: 0, 465: 12, 466: 58, 467: 9, 468: 63, 469: 14, 470: 56, 471: 64, 472: 7, 473: 56, 474: 12, 475: 55, 476: 9, 477: 56, 478: 13, 479: 55, 480: 55, 481: 55, 482: 22, 483: 9, 484: 7, 485: 22, 486: 34, 487: 22, 488: 56, 489: 23, 490: 12, 491: 56, 492: 13, 493: 31, 494: 34, 495: 31, 496: 18, 497: 9, 498: 9, 499: 14, 500: 9, 501: 12, 502: 12, 503: 14, 504: 14, 505: 14, 506: 55, 507: 55, 508: 22, 509: 9, 510: 7, 511: 63, 512: 14, 513: 34, 514: 12, 515: 12, 516: 31, 517: 63, 518: 32, 519: 13, 520: 31, 521: 14, 522: 4, 523: 56, 524: 12, 525: 43, 526: 31, 527: 22, 528: 22, 529: 12, 530: 22, 531: 22, 532: 31, 533: 18, 534: 22, 535: 55, 536: 43, 537: 63, 538: 9, 539: 18, 540: 43, 541: 34, 542: 34, 543: 56, 544: 14, 545: 22, 546: 34, 547: 58, 548: 31, 549: 44, 550: 14, 551: 42, 552: 0, 553: 31, 554: 7, 555: 63, 556: 31, 557: 43, 558: 34, 559: 31, 560: 54, 561: 63, 562: 43, 563: 56, 564: 31, 565: 58, 566: 34, 567: 14, 568: 12, 569: 63, 570: 0, 571: 55, 572: 14, 573: 63, 574: 4, 575: 63, 576: 7, 577: 34, 578: 12, 579: 34, 580: 9, 581: 63, 582: 9, 583: 64, 584: 0, 585: 42, 586: 63, 587: 56, 588: 18, 589: 22, 590: 22, 591: 0, 592: 22, 593: 34, 594: 34, 595: 63, 596: 56, 597: 64, 598: 9, 599: 42, 600: 56, 601: 12, 602: 54, 603: 63, 604: 55, 605: 22, 606: 56, 607: 29, 608: 12, 609: 63, 610: 12, 611: 44, 612: 63, 613: 22, 614: 12, 615: 12, 616: 42, 617: 12, 618: 14, 619: 18, 620: 22, 621: 63, 622: 55, 623: 56, 624: 9, 625: 7, 626: 56, 627: 63, 628: 7, 629: 0, 630: 12, 631: 42, 632: 22, 633: 56, 634: 9, 635: 56, 636: 0, 637: 31, 638: 12, 639: 12, 640: 43, 641: 34, 642: 34, 643: 12, 644: 56, 645: 43, 646: 43, 647: 14, 648: 13, 649: 43, 650: 22, 651: 22, 652: 12, 653: 13, 654: 63, 655: 12, 656: 63, 657: 64, 658: 12, 659: 14, 660: 9, 661: 63, 662: 22, 663: 9, 664: 22, 665: 63, 666: 14, 667: 32, 668: 9, 669: 42, 670: 63, 671: 63, 672: 9, 673: 49, 674: 55, 675: 63, 676: 42, 677: 56, 678: 0, 679: 0, 680: 42, 681: 44, 682: 9, 683: 34, 684: 34, 685: 22, 686: 55, 687: 34, 688: 22, 689: 12, 690: 63, 691: 12, 692: 13, 693: 54, 694: 7, 695: 55, 696: 56, 697: 12, 698: 9, 699: 34, 700: 44, 701: 1, 702: 54, 703: 31, 704: 22, 705: 63, 706: 9, 707: 22, 708: 31, 709: 42, 710: 56, 711: 42, 712: 35, 713: 22, 714: 34, 715: 32, 716: 23, 717: 63, 718: 43, 719: 18, 720: 13, 721: 18, 722: 4, 723: 57, 724: 45, 725: 14, 726: 56, 727: 9, 728: 13, 729: 18, 730: 63, 731: 56, 732: 22, 733: 54, 734: 63, 735: 12, 736: 31, 737: 27, 738: 14, 739: 56, 740: 22, 741: 18, 742: 22, 743: 9, 744: 64, 745: 22, 746: 54, 747: 54, 748: 0, 749: 56, 750: 18, 751: 63, 752: 54, 753: 22, 754: 22, 755: 55, 756: 13, 757: 63, 758: 55, 759: 22, 760: 31, 761: 22, 762: 9, 763: 64, 764: 64, 765: 31, 766: 14, 767: 56, 768: 4, 769: 56, 770: 12, 771: 13, 772: 56, 773: 27, 774: 12, 775: 12, 776: 34, 777: 64, 778: 22, 779: 63, 780: 7, 781: 54, 782: 18, 783: 56, 784: 56, 785: 55, 786: 55, 787: 56, 788: 9, 789: 31, 790: 13, 791: 13, 792: 56, 793: 32, 794: 18, 795: 54, 796: 12, 797: 0, 798: 55, 799: 31, 800: 22, 801: 54, 802: 63, 803: 63, 804: 18, 805: 4, 806: 12, 807: 55, 808: 32, 809: 14, 810: 22, 811: 22, 812: 1, 813: 14, 814: 7, 815: 43, 816: 55, 817: 63, 818: 22, 819: 31, 820: 58, 821: 43, 822: 34, 823: 56, 824: 0, 825: 43, 826: 22, 827: 31, 828: 14, 829: 63, 830: 31, 831: 31, 832: 9, 833: 7, 834: 12, 835: 43, 836: 56, 837: 56, 838: 42, 839: 43, 840: 56, 841: 12, 842: 12, 843: 57, 844: 22, 845: 56, 846: 18, 847: 63, 848: 22, 849: 14, 850: 57, 851: 22, 852: 4, 853: 9, 854: 18, 855: 56, 856: 63, 857: 31, 858: 9, 859: 22, 860: 9, 861: 31, 862: 56, 863: 43, 864: 63, 865: 9, 866: 63, 867: 63, 868: 18, 869: 12, 870: 63, 871: 7, 872: 55, 873: 43, 874: 63, 875: 34, 876: 31, 877: 55, 878: 22, 879: 0, 880: 63, 881: 34, 882: 22, 883: 18, 884: 13, 885: 42, 886: 55, 887: 12, 888: 43, 889: 34, 890: 4, 891: 22, 892: 55, 893: 0, 894: 31, 895: 1, 896: 13, 897: 22, 898: 13, 899: 13, 900: 43, 901: 13, 902: 56, 903: 12, 904: 18, 905: 18, 906: 0, 907: 27, 908: 42, 909: 14, 910: 14, 911: 42, 912: 23, 913: 7, 914: 7, 915: 9, 916: 55, 917: 44, 918: 44, 919: 43, 920: 22, 921: 44, 922: 44, 923: 14, 924: 27, 925: 27, 926: 14, 927: 27, 928: 27, 929: 27, 930: 27, 931: 27, 932: 27, 933: 27, 934: 27, 935: 27, 936: 62, 937: 62, 938: 62, 939: 29, 940: 29, 941: 29, 942: 62, 943: 62, 944: 62, 945: 62, 946: 26, 947: 30, 948: 29, 949: 29, 950: 29, 951: 29, 952: 29, 953: 29, 954: 29, 955: 29, 956: 29, 957: 29, 958: 42, 959: 27, 960: 27, 961: 27, 962: 27, 963: 27, 964: 27, 965: 27, 966: 27, 967: 27, 968: 14, 969: 27, 970: 43, 971: 42, 972: 43, 973: 15, 974: 43, 975: 43, 976: 43, 977: 43, 978: 43, 979: 43, 980: 43, 981: 45, 982: 45, 983: 45, 984: 26, 985: 26, 986: 26, 987: 27, 988: 46, 989: 26, 990: 46, 991: 15, 992: 30, 993: 30, 994: 30, 995: 30, 996: 30, 997: 30, 998: 27, 999: 42}
        #self.labels = self.labels.map(self.dict0)
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.transform(Image.open(os.path.join(self.root, self.images[idx])))
        label = self.labels[idx]
        return image, label

def get_eval_test_dataloader(options, processor):
    if(options.eval_test_data_dir is None): return

    if(options.eval_data_type == "Caltech101"):
        dataset = ImageLabelDataset(root = options.eval_test_data_dir, transform = processor.process_image)
    elif(options.eval_data_type == "CIFAR10"):
        dataset = torchvision.datasets.CIFAR10(root = os.path.dirname(options.eval_test_data_dir), download = True, train = False, transform = processor.process_image)
    elif(options.eval_data_type == "CIFAR100"):
        dataset = torchvision.datasets.CIFAR100(root = os.path.dirname(options.eval_test_data_dir), download = True, train = False, transform = processor.process_image)
        #dataset.targets = sparse2coarse(dataset.targets)
    elif(options.eval_data_type == "DTD"):
        dataset = torchvision.datasets.DTD(root = os.path.dirname(options.eval_test_data_dir), download = True, split = "test", transform = processor.process_image)
    elif(options.eval_data_type == "FGVCAircraft"):
        dataset = torchvision.datasets.FGVCAircraft(root = os.path.dirname(options.eval_test_data_dir), download = True, split = "test", transform = processor.process_image)
    elif(options.eval_data_type == "Flowers102"):
        dataset = torchvision.datasets.Flowers102(root = os.path.dirname(options.eval_test_data_dir), download = True, split = "test", transform = processor.process_image)
    elif(options.eval_data_type == "Food101"):
        dataset = torchvision.datasets.Food101(root = os.path.dirname(options.eval_test_data_dir), download = True, split = "test", transform = processor.process_image)
    elif(options.eval_data_type == "GTSRB"):
        dataset = torchvision.datasets.GTSRB(root = os.path.dirname(options.eval_test_data_dir), download = True, split = "test", transform = processor.process_image)
    elif(options.eval_data_type == "ImageNet1K"):
        dataset = ImageLabelDataset(root = options.eval_test_data_dir, transform = processor.process_image)
    elif(options.eval_data_type == "OxfordIIITPet"):
        dataset = torchvision.datasets.OxfordIIITPet(root = os.path.dirname(options.eval_test_data_dir), download = True, split = "test", transform = processor.process_image)
    elif(options.eval_data_type == "coco"):
        dataset = ImageLabelDataset(root = options.eval_test_data_dir, transform = processor.process_image)
    elif(options.eval_data_type == "RenderedSST2"):
        dataset = torchvision.datasets.RenderedSST2(root = os.path.dirname(options.eval_test_data_dir), download = True, split = "test", transform = processor.process_image)
    elif(options.eval_data_type == "StanfordCars"):
        dataset = torchvision.datasets.StanfordCars(root = os.path.dirname(options.eval_test_data_dir), download = True, split = "test", transform = processor.process_image)
    elif(options.eval_data_type == "STL10"):
        dataset = torchvision.datasets.STL10(root = os.path.dirname(options.eval_test_data_dir), download = True, split = "test", transform = processor.process_image)
    elif(options.eval_data_type == "SVHN"):
        dataset = torchvision.datasets.SVHN(root = os.path.dirname(options.eval_test_data_dir), download = True, split = "test", transform = processor.process_image)
    elif(options.eval_data_type in ["ImageNetSketch", "ImageNetV2", "ImageNet-A", "ImageNet-R"]):
        dataset = ImageLabelDataset(root = options.eval_test_data_dir, transform = processor.process_image)
    else:
        raise Exception(f"Eval test dataset type {options.eval_data_type} is not supported")
    dataloader = torch.utils.data.DataLoader(dataset, batch_size = options.batch_size, num_workers = options.num_workers, sampler = None)
    dataloader.num_samples = len(dataset)
    dataloader.num_batches = len(dataloader)

    return dataloader

def get_eval_train_dataloader(options, processor):
    if(not options.linear_probe or options.eval_train_data_dir is None): return

    if(options.eval_data_type == "Caltech101"):
        dataset = ImageLabelDataset(root = options.eval_train_data_dir, transform = processor.process_image)
    elif(options.eval_data_type == "CIFAR10"):
        dataset = torchvision.datasets.CIFAR10(root = os.path.dirname(options.eval_train_data_dir), download = True, train = True, transform = processor.process_image)
    elif(options.eval_data_type == "CIFAR100"):
        dataset = torchvision.datasets.CIFAR100(root = os.path.dirname(options.eval_test_data_dir), download = True, train = True, transform = processor.process_image)
        #dataset.targets = sparse2coarse(dataset.targets)
    elif(options.eval_data_type == "DTD"):
        dataset = torch.utils.data.ConcatDataset([torchvision.datasets.DTD(root = os.path.dirname(options.eval_train_data_dir), download = True, split = "train", transform = processor.process_image), torchvision.datasets.DTD(root = os.path.dirname(options.eval_train_data_dir), download = True, split = "val", transform = processor.process_image)])
    elif(options.eval_data_type == "FGVCAircraft"):
        dataset = torchvision.datasets.FGVCAircraft(root = os.path.dirname(options.eval_train_data_dir), download = True, split = "trainval", transform = processor.process_image)
    elif(options.eval_data_type == "Flowers102"):
        dataset = torchvision.datasets.Flowers102(root = os.path.dirname(options.eval_test_data_dir), download = True, split = "test", transform = processor.process_image)
    elif(options.eval_data_type == "Food101"):
        dataset = torchvision.datasets.Food101(root = os.path.dirname(options.eval_train_data_dir), download = True, split = "train", transform = processor.process_image)
    elif(options.eval_data_type == "GTSRB"):
        dataset = torchvision.datasets.GTSRB(root = os.path.dirname(options.eval_train_data_dir), download = True, split = "train", transform = processor.process_image)
    elif(options.eval_data_type == "ImageNet1K"):
        dataset = ImageLabelDataset(root = options.eval_train_data_dir, transform = processor.process_image)
    elif(options.eval_data_type == "OxfordIIITPet"):
        dataset = torchvision.datasets.OxfordIIITPet(root = os.path.dirname(options.eval_train_data_dir), download = True, split = "trainval", transform = processor.process_image)
    elif(options.eval_data_type == "RenderedSST2"):
        dataset = torchvision.datasets.RenderedSST2(root = os.path.dirname(options.eval_train_data_dir), download = True, split = "train", transform = processor.process_image)
    elif(options.eval_data_type == "StanfordCars"):
        dataset = torchvision.datasets.StanfordCars(root = os.path.dirname(options.eval_train_data_dir), download = True, split = "train", transform = processor.process_image)
    elif(options.eval_data_type == "STL10"):
        dataset = torchvision.datasets.STL10(root = os.path.dirname(options.eval_train_data_dir), download = True, split = "train", transform = processor.process_image)
    elif(options.eval_data_type == "SVHN"):
        dataset = torchvision.datasets.SVHN(root = os.path.dirname(options.eval_train_data_dir), download = True, split = "train", transform = processor.process_image)
    else:
        raise Exception(f"Eval train dataset type {options.eval_data_type} is not supported")

    dataloader = torch.utils.data.DataLoader(dataset, batch_size = options.linear_probe_batch_size, num_workers = options.num_workers, sampler = None)
    dataloader.num_samples = len(dataset)
    dataloader.num_batches = len(dataloader)

    return dataloader

def load(options, processor):
    data = {}
    
    data["train"] = get_train_dataloader(options, processor)
    data["validation"] = get_validation_dataloader(options, processor)
    data["eval_test"] = get_eval_test_dataloader(options, processor)
    data["eval_train"] = get_eval_train_dataloader(options, processor)

    return data

def load_eval(options, processor, data):
    
    if(not options.linear_probe):
        data["eval_test"] = get_eval_test_dataloader(options, processor)
    data["eval_train"] = get_eval_train_dataloader(options, processor)

    return data