from segmentall-base:latest
WORKDIR "/app"
COPY . .
CMD ["bash","/app/run_services.sh"]
