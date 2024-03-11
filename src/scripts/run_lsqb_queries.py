import os
import sys

sys.path.append(os.getcwd())

from src.query.query_sparql import execute_sparql


def main():

    for filename in os.listdir("data/queries/lsqb/"):
        print(f"running query: {filename}")
        with open(f"data/queries/lsqb/{filename}") as f:
            query = f.read()
            res, time = execute_sparql(query=query)
            print(time)
    # with open("data/queries/lsqb/q1.sparql") as f:
    #     query = f.read()
    # get all the slow timeouted queries from old run
    # df_res = pd.read_csv(res_path + f'results_{qtype.value}.csv')
    # df_res = df_res[df_res['exec_time'] == 0]


    # df = pd.read_csv(f'data/queries/wdbench/{qtype.value}.txt', header=None)
    # df.rename(columns={0: 'id', 1: 'query_parts'}, inplace=True)

    # df = df[df['id'].isin(df_res['query_id'])]
    # print(50*'-')
    # print(qtype.value)
    # print(50*'-')
    # run_all_in_df(query_df=df, query_type=qtype.value)
        
    #     res, time = execute_sparql(query=query)
    
    # print(time)



if __name__ == "__main__":
    main()