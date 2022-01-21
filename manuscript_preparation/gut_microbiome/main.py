import pandas as pd


TOTAL_NUM_SAMPLES = 94342


def main():
    # lrp
    df_lrp = pd.read_csv(
        './lrp.txt',
        sep='\t',
        names=['mark', 'target', 'run_sample_ids', 'e-value'])

    run_sample_ids = df_lrp['run_sample_ids'].tolist()
    run_sample_ids = [y for x in run_sample_ids for y in x.split(' ')]
    run_sample_ids = list(set(run_sample_ids))
    lrp_percentage = len(run_sample_ids) * 100 / TOTAL_NUM_SAMPLES
    print(f'Number of unique samples for lrp: {len(run_sample_ids)}/{TOTAL_NUM_SAMPLES} ({lrp_percentage:.2f}%)')

    #rbsK
    df_rbsK = pd.read_csv(
        './rbsK.txt',
        sep='\t',
        names=['mark', 'target', 'run_sample_ids', 'e-value'])

    run_sample_ids = df_rbsK['run_sample_ids'].tolist()
    run_sample_ids = [y for x in run_sample_ids for y in x.split(' ')]
    run_sample_ids = list(set(run_sample_ids))
    rbsK_percentage = len(run_sample_ids) * 100 / TOTAL_NUM_SAMPLES
    print(f'Number of unique samples for rbsK: {len(run_sample_ids)}/{TOTAL_NUM_SAMPLES} ({rbsK_percentage:.2f}%)')

    #qorB
    df_qorB = pd.read_csv(
        './qorB.txt',
        sep='\t',
        names=['mark', 'target', 'run_sample_ids', 'e-value'])

    run_sample_ids = df_qorB['run_sample_ids'].tolist()
    run_sample_ids = [y for x in run_sample_ids for y in x.split(' ')]
    run_sample_ids = list(set(run_sample_ids))
    qorB_percentage = len(run_sample_ids) * 100 / TOTAL_NUM_SAMPLES
    print(f'Number of unique samples for qorB: {len(run_sample_ids)}/{TOTAL_NUM_SAMPLES} ({qorB_percentage:.2f}%)')

    #hdfR
    df_hdfR = pd.read_csv(
        './hdfR.txt',
        sep='\t',
        names=['mark', 'target', 'run_sample_ids', 'e-value'])

    run_sample_ids = df_hdfR['run_sample_ids'].tolist()
    run_sample_ids = [y for x in run_sample_ids for y in x.split(' ')]
    run_sample_ids = list(set(run_sample_ids))
    hdfR_percentage = len(run_sample_ids) * 100 / TOTAL_NUM_SAMPLES
    print(f'Number of unique samples for hdfR: {len(run_sample_ids)}/{TOTAL_NUM_SAMPLES} ({hdfR_percentage:.2f}%)')

    #ftsP
    df_ftsP = pd.read_csv(
        './ftsP.txt',
        sep='\t',
        names=['mark', 'target', 'run_sample_ids', 'e-value'])

    run_sample_ids = df_ftsP['run_sample_ids'].tolist()
    run_sample_ids = [y for x in run_sample_ids for y in x.split(' ')]
    run_sample_ids = list(set(run_sample_ids))
    ftsP_percentage = len(run_sample_ids) * 100 / TOTAL_NUM_SAMPLES
    print(f'Number of unique samples for ftsP: {len(run_sample_ids)}/{TOTAL_NUM_SAMPLES} ({ftsP_percentage:.2f}%)')

    #proV



if __name__ == '__main__':
    main()