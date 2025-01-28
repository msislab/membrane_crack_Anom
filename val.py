from ultralytics import YOLO
import argparse
import torch

def validate():
    args = parseArgs()
    # choose between training from scratch and training using a checkpoint
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    # Load YOLO model    
    model = YOLO(args.model)

    # Train model
    model.val(
        data    = args.data       ,          
        batch   = args.batch_size ,         
        imgsz   = args.img_size   ,                           
        project = args.project    ,       
        device  = device          ,
        save_txt= args.save_txt   ,
        conf    = args.conf       ,
        verbose = True       
    )

def parseArgs():
    parser = argparse.ArgumentParser(description='Yolov8 validation'                                       )
    parser.add_argument("--model"      , type=str, default='',
                        help='provide a model path (model.pt) if initializing from a pretrained checkpoint')
    parser.add_argument("--data"       , type=str, default='data.yaml', help='provide data.yaml path'      )
    parser.add_argument("--batch_size" , type=int, default=16, help='specify the batch size'               )
    parser.add_argument("--img_size"   , type=int, default=832, help='specify the image size'              )
    parser.add_argument("--project"    , type=str, default='runs/Yolov8_Tutorial', help='name your project')
    parser.add_argument("--name"       , type=str, default='custom_training'                               )
    parser.add_argument("--device"     , type=int, default=0, help='Chosse a gpu device'                   )
    parser.add_argument("--save_txt"   , action='store_true'                                               )
    parser.add_argument("--conf"       , type=float, default=0.1                                          )

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    # Run training
    validate()