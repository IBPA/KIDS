
# # process Girgis et al.
# echo 'Processing Girgis et al...'
# cd Girgis_et_al
# python3 create_data.py
# cd ..

# # process GO
# echo 'Processing GO...'
# cd GO
# python3 create_data.py
# cd ..

# # process Shaw et al.
# echo 'Processing Shaw et al...'
# cd Shaw_et_al
# python3 create_data.py
# cd ..
# python3 ensure_official_gene_symbols.py ./Shaw_et_al/Shaw_et_al_intermediate.txt ./Shaw_et_al/Shaw_et_al.txt 'upregulated by antibiotic after 30 mins' 'not upregulated by antibiotic after 30 mins'

# # process Soo et al.
# echo 'Processing Soo et al...'
# cd Soo_et_al
# python3 create_data.py
# cd ..

# # process Zhou et al.
# echo 'Processing Zhou et al...'
# cd Zhou_et_al
# python3 create_data.py
# cd ..
