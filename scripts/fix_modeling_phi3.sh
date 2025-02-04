sed -i 's/get_max_length/get_max_cache_shape/g' .cache_hf/hub/models--microsoft--Phi-*/snapshots/*/modeling_phi3.py
cat .cache_hf/hub/models--microsoft--Phi-*/snapshots/*/modeling_phi3.py | grep get_max_
