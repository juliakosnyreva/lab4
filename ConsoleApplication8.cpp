#include <iostream>
#include <random>
#include "mpi.h"
#include <chrono>

#define N 4 // константа N для размерности квадратных матриц

using namespace std;

void print_matrix(int m[N][N], string name) {
    cout << name << " = " << endl;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            cout << " " << m[i][j] << " ";
        }
        cout << endl;
    }
    cout << endl;
}

int main(int argc, char* argv[]) {
    int rank, numtasks;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);

    // генерация случ чисел
    random_device rd;
    mt19937 gen(rd() + rank);
    uniform_int_distribution<int> dis(1, 1000);

    int a[N][N];
    int b[N][N];
    int c[N][N];
    int aa[N], cc[N];

    MPI_Barrier(MPI_COMM_WORLD);

    // заполнение случ числами
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            a[i][j] = dis(gen);
            b[i][j] = dis(gen);
        }
    }

    MPI_Scatter(a, N * N / numtasks, MPI_INT, aa, N * N / numtasks, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(b, N * N, MPI_INT, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);

    auto start = chrono::steady_clock::now();

    //перемножение матриц
    int sum = 0;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            sum += aa[j] * b[j][i];
        }
        cc[i] = sum;
        sum = 0;
    }

    MPI_Gather(cc, N * N / numtasks, MPI_INT, c, N * N / numtasks, MPI_INT, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);

    auto end = chrono::steady_clock::now();
    if (rank == 0) {
        auto duration = chrono::duration_cast<chrono::duration<double>>(end - start);
        cout << "time:  " << duration.count() << " seconds" << endl;
        print_matrix(a, "A");
        print_matrix(b, "B");
        print_matrix(c, "C");
    }

    MPI_Finalize();
}
