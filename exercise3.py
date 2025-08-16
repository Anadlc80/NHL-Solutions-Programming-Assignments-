# I found the following mistakes in the code:
# 1)The laber of the coordanates are mixed, I changed them x= Presicion and y= Recall
# 2)They were using a 2D array of String when they should be float, I converted its type using map to convert every elem and then list to conver it in a list.
# POSSIBLE IMPROVEMENTS: 
# 1) The function didin't control any possible mistake in the csv file, because the missed of the header --> (None)
# 2) We would need to considerate also the lenght of each row ( the user could introduce row of diffenet lenghts) to be able to use np.stack else it would return a error message
# I dind't do it bucause I'm not sure about how much I can change the code, but I think we need a vble "longitud" and check every loop the currently lenght is the same of the nexr row.
# 3)I added some Matplotlib functions for the graphin: plt.grid(True) --> show the gride and plt.tight_layout --> margins
# 4)On Windows, newline="" is used to prevent extra blank lines in the CSV.


def plot_data(csv_file_path: str):
    results = []
    with open(csv_file_path) as result_csv:
        csv_reader = csv.reader(result_csv, delimiter=',')
        # I check is the file doesn't have a header
        next(csv_reader, None)
        for row in csv_reader: 
            # Here we should check the lenght of each row, with a variable inicializate outside of the loop
            fila_float=list(map(float,row))
            results.append(fila_float)
        results = np.stack(results,dtype=float)

    # plot precision-recall curve
    plt.plot(results[:, 1], results[:, 0])
    plt.ylim([-0.05, 1.05])
    plt.xlim([-0.05, 1.05])
    plt.xlabel('Precision')
    plt.ylabel('Recall')
    #Im going to improve the quality of the graphic
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# newline="" is used to prevent extra blank lines in the CSV on Windowns enviroments.
f = open("data_file.csv", "w",newline="")
w = csv.writer(f)
_ = w.writerow(["precision", "recall"])
w.writerows([[0.013,0.951],
             [0.376,0.851],
             [0.441,0.839],
             [0.570,0.758],
             [0.635,0.674],
             [0.721,0.604],
             [0.837,0.531],
             [0.860,0.453],
             [0.962,0.348],
             [0.982,0.273],
             [1.0,0.0]])
f.close()
plot_data('data_file.csv')
