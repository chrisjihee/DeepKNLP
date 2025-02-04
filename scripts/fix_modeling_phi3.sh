sed -i 's/get_max_length/get_max_cache_shape/g' .cache_hf/*/*/*/*/*.py
sed -i 's/get_max_length/get_max_cache_shape/g' .cache_hf/*/*/*/*/*/*.py
grep -r get_max_ .cache_hf/*/*/*/*/*.py .cache_hf/*/*/*/*/*/*.py
