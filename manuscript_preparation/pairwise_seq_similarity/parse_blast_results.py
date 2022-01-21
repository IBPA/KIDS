import sys
import pandas as pd


def main():
    df_arg = pd.read_csv('./blast_result_arg_card.txt', sep='\t', names=list(range(12)))
    df_no_arg = pd.read_csv('./blast_result_no_arg_card.txt', sep='\t', names=list(range(12)))

    df_arg.drop_duplicates(subset=0, keep='first', inplace=True, ignore_index=True)
    df_arg = df_arg[[0, 1, 2, 10]]

    df_no_arg.drop_duplicates(subset=0, keep='first', inplace=True, ignore_index=True)
    df_no_arg = df_no_arg[[0, 1, 2, 10]]

    df_arg.to_csv('./blast_arg_parsed.txt', sep='\t', index=False, header=False)
    df_no_arg.to_csv('./blast_no_arg_parsed.txt', sep='\t', index=False, header=False)


if __name__ == '__main__':
    main()
