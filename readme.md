运行docker,使用NeuralHDHair的镜像
docker run --gpus all --name segmentall-server -d  -p 50083:50083 -w /app -v "$(pwd):/app" hair-base    bash /app/run_services.sh

重新运行docker
docker restart segmentall-server