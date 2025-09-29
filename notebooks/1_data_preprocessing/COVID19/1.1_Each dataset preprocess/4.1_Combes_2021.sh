input_dir="/data/wuqinhua/phase/covid19/datasets/pre_data/4_Combes_2021"
filter_dir="$input_dir/filter_data"
output_dir="$input_dir/data"

# Create output directory if it doesn't exist
mkdir -p "$output_dir"

# Move and organize files
for file in "$filter_dir"/*.gz; do
    if [ -f "$file" ]; then
        base_name=$(basename "$file")
        folder_name="${base_name:0:10}"
        folder_path="$output_dir/$folder_name"

        mkdir -p "$folder_path"
        cp "$file" "$folder_path/"
    fi
done

# Organize files within subdirectories
for dir in "$output_dir"/*; do
    if [ -d "$dir" ]; then
        find "$dir" -type f -name "*barcodes.tsv.gz" -execdir mv {} "$dir/barcodes.tsv.gz" \;
        find "$dir" -type f -name "*features.tsv.gz" -execdir mv {} "$dir/features.tsv.gz" \;
        find "$dir" -type f -name "*matrix.mtx.gz" -execdir mv {} "$dir/matrix.mtx.gz" \;
    fi
done
