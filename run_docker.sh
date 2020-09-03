sudo docker run --net host --gpus all --rm -it --name img1 -v $PWD:/workspace -p 8001:8001 -p 8002:8002 -p 8003:8003 -w /workspace yxlin/darknet:gpu-cv-cc75 bash
