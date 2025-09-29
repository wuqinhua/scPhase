data_directory="/data/wuqinhua/phase/covid19/datasets/pre_data/16_Unterman_2022/data"

for file in "$data_directory"/*.tar.gz; do
  if [ -f "$file" ]; then
    filename=$(basename "$file" .tar.gz)
    mkdir -p "$data_directory/$filename"
    tar -xzf "$file" -C "$data_directory/$filename"
    echo "已解压文件: $file 到目录: $data_directory/$filename"
  fi
done