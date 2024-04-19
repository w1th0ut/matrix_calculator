import numpy as np
from typing import Union
from numpy import array, zeros, fabs, linalg
import re
import os

# =========================================================================================================================

# Akmal Yusuf                  L0123014 
# Aufa Rafly Ferdiansyah 	   L0123031 
# Bagas Rizki Gunardi 	       L0123034 

# ======================================================== MENU ===========================================================

def menu():
    print("=========================================")
    print("         MENU KALKULATOR MATRIKS         ")
    print("=========================================")
    print("Pilih Operasi Progam :")
    print("1. Sistem Persamaaan Linier ")
    print("2. Determinan")
    print("3. Matriks Invers")
    print("4. LU Faktorisasi")
    print("5. Persamaan Polinomial, Nilai Eigen, vector eigen")
    print("6. Diagonalisasi Matriks")
    print("7. Singular Value Dekomposision (SVD)")
    print("8. Keluar")

# ====================================================== SPL SECTION =======================================================

def spl_gaus_complex():
    n = int(input("Masukkan jumlah baris: "))
    m = int(input("Masukkan jumlah kolom: "))
    print(f"Masukkan koefisien matriks A (ukuran {n}x{m}):")
    A = []
    for _ in range(n):
        row = [np.complex_(x) for x in input().split()]
        A.append(row)
    A = np.array(A)
    matrix1 = str(A)
    print(f"Masukkan vektor b (dengan ukuran {n}):")
    B = np.array([np.complex_(x) for x in input().split()])
    matrix2 = str(B)
    def splgaus(A, b):
        n = len(b)
        # Elimination phase
        for k in range(0, n-1):
            for i in range(k+1, n):
                if A[i, k] != 0.0:
                    lam = A[i, k] / A[k, k]
                    A[i, k+1:n] = A[i, k+1:n] - lam * A[k, k+1:n]
                    b[i] = b[i] - lam * b[k]

        # Back-substitution
        x = np.zeros_like(b, dtype=complex)
        for k in range(n-1, -1, -1):
            x[k] = (b[k] - np.dot(A[k, k+1:n], x[k+1:n])) / A[k, k]
        return x

    solution = splgaus(A, B)

    augmented_matrix = np.hstack((A, B.reshape(-1, 1)))
    temp_matrix = np.linalg.matrix_rank(augmented_matrix)
    cols = A.shape[1]

    if solution is None:
        print("\nOutput: \n(No Solution)")
    elif temp_matrix < cols:
        print("\nOutput: \n(Infinite Solutions)")
    else:
        print("\nOutput: \n(Unique Solution)")
        for i in range(n):
            print(f"x{i+1} = {np.round(solution[i])}")
    
    with open('hasil.txt', 'a') as file:
        file.write("Soal SPL Kompleks dengan Gauss:\n")
        file.write("Matriks A =\n")
        file.write(f"{matrix1}\n\n")
        file.write("Matriks B =\n")
        file.write(f"{matrix2}\n\n")
        file.write("Output :\n")
        for i in range(n):
            file.write(f"x{i+1} = {np.round(solution[i])}\n")
        file.write("\n")

def spl_gaus():
    try:
        n = int(input("Masukkan jumlah baris: "))
        m = int(input("Masukkan jumlah kolom: "))
        print(f"Masukkan koefisien matriks A (ukuran {n}x{m}):")
        A = []
        for _ in range(n):
            row = [float(x) for x in input().split()]
            A.append(row)
        A = np.array(A)
        matrix1 = str(A)

        print(f"Masukkan vektor b (ukuran {n}):")
        b = np.array([float(x) for x in input().split()])
        matrix2 = str(b)

        x = np.zeros(n, float)

        for k in range(n - 1):
            if abs(A[k, k]) < 1.0e-12:
                for i in range(k + 1, n):
                    if abs(A[i, k]) > abs(A[k, k]):
                        A[[k, i]] = A[[i, k]]
                        b[[k, i]] = b[[i, k]]
                        break

            for i in range(k + 1, n):
                if A[i, k] == 0:
                    continue
                factor = A[i, k] / A[k, k]
                for j in range(k, n):
                    A[i, j] = A[i, j] - A[k, j] * factor
                b[i] = b[i] - b[k] * factor

        if abs(A[n - 1, n - 1]) < 1.0e-12:
            x[n - 1] = 0  
        else:
            x[n - 1] = b[n - 1] / A[n - 1, n - 1]
        
        # Menghitung rank matriks A dan matriks augmented
        rank_a = np.linalg.matrix_rank(A)
        augmented_matrix = np.hstack((A, b.reshape(-1, 1)))
        rank_aug = np.linalg.matrix_rank(augmented_matrix)

        if rank_a != rank_aug:
            print("\nOutput: \n(No Solution)")
            result = "No Solution\n"
        elif rank_a == rank_aug and rank_a < len(np.transpose(A)):
            print("\nOutput: \n(Infinite Solutions)")
            result = "Infinite Solutions\n"
            free_parameters = set(range(1, len(np.transpose(A)) + 1))
            free_parameters -= set(np.argmax(A != 0, axis=1) + 1)
            print("Free Parameters:", free_parameters)
            for param in free_parameters:
                print(f"x{param} is a free parameter")
                result += f"x{param} is a free parameter\n"
        else:
            print("\nOutput: \n(Unique Solution)")
            result = "Unique Solution:\n"
            for k in range(n - 1, -1, -1):
                sum_ax = np.dot(A[k, k+1:], x[k+1:])
                x[k] = (b[k] - sum_ax) / A[k, k]
                print(f"x{k + 1} = {x[k]}")
                result += f"x{k + 1} = {x[k]}\n"

        with open('hasil.txt', 'a') as file:
            file.write("Soal SPL Pakai Gauss:\n")
            file.write("Matriks A =\n")
            file.write(matrix1 + "\n\n")
            file.write("Matriks B =\n")
            file.write(matrix2 + "\n\n")
            file.write("Output :\n")
            file.write(result + "\n")

    except ValueError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Error: {e}")

def pilihan_gaus():
    print("=========================================")
    print("            Gauss Elimination            ")
    print("=========================================")
    print("Pilih Jenis Matrix: ")
    print("1. Matrix Biasa")
    print("2. Matrix Kompleks")

    a = int(input("Pilihan anda: "))
    match a:
        case 1:
            spl_gaus()
        case 2: 
            spl_gaus_complex()
        case __:
            print("Pilihan tidak valid")

def splgaus_Jordan(A, b):
    n = len(A)
    
    for i in range(n):
        if A[i][i] == 0:
            return None
        
        for j in range(i+1, n):
            ratio = A[j][i] / A[i][i]
            
            for k in range(n):
                A[j][k] -= ratio * A[i][k]
            
            b[j] -= ratio * b[i]
    
    x = np.zeros(n)
    x[n-1] = b[n-1] / A[n-1][n-1]
    
    for i in range(n-2, -1, -1):
        x[i] = b[i]
        
        for j in range(i+1, n):
            x[i] -= A[i][j] * x[j]
        
        x[i] /= A[i][i]
    
    return x

def spl_Jordan():
    n = int(input("Masukkan jumlah baris: "))
    m = int(input("Masukkan jumlah kolom: "))

    print(f"Masukkan koefisien matriks A (ukuran {n}x{m}):")
    A = []
    for _ in range(n):
        row = [float(x) for x in input().split()]
        A.append(row)
    A = np.array(A)
    matrix1 = str(A)
    print(f"Masukkan vektor b (ukuran {n}):")
    B = np.array([float(x) for x in input().split()])
    matrix2 = str(B)
    solution = splgaus_Jordan(A, B)

    augmented_matrix = np.hstack((A, B.reshape(-1, 1)))
    rank_a = np.linalg.matrix_rank(A)
    rank_aug = np.linalg.matrix_rank(augmented_matrix)
    

    if rank_a != rank_aug:
        print("\nOutput: \n(No Solution)")
        result = "No Solution\n"
    elif rank_a == rank_aug and rank_a < len(np.transpose(A)):
        print("\nOutput: \n(Infinite Solutions)")
        result = "Infinite Solutions\n"
        free_parameters = set(range(1, len(np.transpose(A)) + 1))
        free_parameters -= set(np.argmax(A != 0, axis=1) + 1)
        print("Free Parameters:", free_parameters)
        for param in free_parameters:
            print(f"x{param} is a free parameter")
            result += f"x{param} is a free parameter\n"
    else:
        print("\nOutput: \n(Unique Solution)")
        result = "Unique Solution:\n"
        for i, sol in enumerate(solution):
            print(f"x{i+1} = {sol}")
            result += (f"x{i+1} = {sol}\n")
    
    with open('hasil.txt', 'a') as file:
        file.write("Soal SPL Pakai Gauss Jordan:\n")
        file.write("Matriks A =\n")
        file.write(matrix1 + "\n\n")
        file.write("Matriks B =\n")
        file.write(matrix2 + "\n\n")
        file.write("Output :\n")
        file.write(result + "\n")

def splBalikan():
    try:
        n = int(input("Masukkan jumlah baris: "))
        m = int(input("Masukkan jumlah kolom: "))

        print(f"Masukkan koefisien matriks A (ukuran {n}x{m}):")
        A = []
        for _ in range(n):
            row = [float(x) for x in input().split()]
            A.append(row)
        A = np.array(A)
        matrix1 = str(A)

        print(f"Masukkan vektor b (ukuran {n}):")
        B = np.array([float(x) for x in input().split()])
        matrix2 = str(B)

        # Cek apakah matriks A memiliki invers
        if np.linalg.det(A) == 0:
            raise ValueError("\nOutput: \n(No Solution)")

        # Jika memiliki invers, cari solusinya
        solution = np.linalg.inv(A) @ B

        augmented_matrix = np.hstack((A, B.reshape(-1, 1)))
        rank_a = np.linalg.matrix_rank(A)
        rank_aug = np.linalg.matrix_rank(augmented_matrix)

        if rank_a != rank_aug:
            print("\nOutput: \n(No Solution)")
            result = "No Solution\n"
        elif rank_a == rank_aug and rank_a < len(np.transpose(A)):
            print("\nOutput: \n(Infinite Solutions)")
            result = "Infinite Solutions\n"
            free_parameters = set(range(1, len(np.transpose(A)) + 1))
            free_parameters -= set(np.argmax(A != 0, axis=1) + 1)
            print("Free Parameters:", free_parameters)
            for param in free_parameters:
                print(f"x{param} is a free parameter")
                result += f"x{param} is a free parameter\n"
        else:
            print("\nOutput: \n(Unique Solution)")
            result = "Unique Solution:\n"
            for i, sol in enumerate(solution):
                print(f"x{i+1} = {sol}")
                result += (f"x{i+1} = {sol}\n")
        
        with open('hasil.txt', 'a') as file:
            file.write("Soal SPL Pakai Invers:\n")
            file.write("Matriks A =\n")
            file.write(matrix1 + "\n\n")
            file.write("Matriks B =\n")
            file.write(matrix2 + "\n\n")
            file.write("Output :\n")
            file.write(result + "\n")

    except ValueError as e:
        print(f"{e}")
    except Exception as e:
        print(f"{e}")

def spl():
    while True:
        print("=========================================")
        print("               SPL SOLVER                ")
        print("=========================================")
        print("Pilih Operasi Progam :")
        print("1. Metode eliminasi Gauss")
        print("2. Metode eliminasi Gauss-Jordan")
        print("3. Metode matriks balikan")
        print("4. Kembali")
        pilih = int(input("Pilihan Anda: "))
        match pilih:
            case 1:
                pilihan_gaus()
            case 2: 
                spl_Jordan()
            case 3:
                splBalikan()
            case 4:
                break
            case _:
                print("Pilihan tidak valid")

# ============================================== LU Faktorisasi Section  =======================================================

def LU():
    print("=========================================")
    print("            LU FACTORIZATION             ")
    print("=========================================")

    try:
        row = int(input("Masukkan jumlah baris matriks: "))
        col = int(input("Masukkan jumlah kolom matriks: "))

        print("Masukkan elemen-elemen matriks baris per baris:")
        matrix = []
        for i in range(row):
            row_elements = list(map(float, input().split()))
            if len(row_elements) != col:
                raise ValueError("Jumlah elemen dalam setiap baris harus sama dengan jumlah kolom matriks.")
            matrix.append(row_elements)

        A = np.array(matrix)
        print("Matriks A:")
        print(A)

        L = np.eye(row)
        U = np.zeros_like(A, dtype=float)

        for i in range(row):
            for k in range(i, row):
                U[i, k] = A[i, k] - np.dot(L[i, :i], U[:i, k])

            for k in range(i + 1, row):
                L[k, i] = (A[k, i] - np.dot(L[k, :i], U[:i, i])) / U[i, i]

        print("Matriks L:")
        print(L)
        print("Matriks U:")
        print(U)

        with open('hasil.txt', 'a') as file:
            file.write("Soal LU Faktorisasi:\n")
            file.write("Matriks A =\n")
            np.savetxt(file, A, fmt='%.2f')
            file.write("\n\n")
            file.write("Matriks L =\n")
            np.savetxt(file, L, fmt='%.2f')
            file.write("\n\n")
            file.write("Matriks U =\n")
            np.savetxt(file, U, fmt='%.2f')
            file.write("\n\n")

    except ValueError as ve:
        print("Error:", ve)
    except Exception as e:
        print("Terjadi kesalahan:", e)
        
# ===================================================== SVD SECTION =======================================================

def svd():
    print("=========================================")
    print("                   SVD                   ")
    print("=========================================")
    n = int(input("Masukkan jumlah baris: "))
    m = int(input("Masukkan jumlah kolom: "))

    print(f"Masukkan koefisien matriks A (ukuran {n}x{m}):")
    A = []
    for _ in range(n):
        row = [float(x) for x in input().split()]
        A.append(row)
    A = np.array(A)

    matrix1 = str(A)
    u, s, vh = np.linalg.svd(A)
    u = np.round(u,4)
    s = np.round(s,4)
    vh = np.round(vh,4)
    s = np.diag(s)

    print("Matriks U:")
    print(u)
    print("Matriks S:")
    print(s)
    print("Matriks Vh:")
    print(vh)
    
    u = str(u)
    s = str(s)
    vh = str(vh)
    with open('hasil.txt', 'a') as file:
        file.write("Soal SVD:\n")
        file.write(matrix1 + "\n\n")
        file.write("Output :\n")
        file.write("Matriks U =\n")
        file.write(u + "\n\n")
        file.write("Matriks S =\n")
        file.write(s + "\n\n")
        file.write("Matriks Vh =\n")
        file.write(vh + "\n\n")

# ============================================== EIGEN VALUES SECTION =======================================================

def hitung_eigen(matriks):
    matriks_np = np.array(matriks)  # Konversi matriks menjadi array NumPy
    eigenvalues, eigenvectors = np.linalg.eig(matriks_np)
    eigenvalues = np.round(eigenvalues, 5)
    eigenvectors = np.round(eigenvectors, 5)
    return eigenvalues, eigenvectors

def hitung_persamaan_polinomial(matriks):
    eigenvalues, _ = hitung_eigen(matriks)
    polynomial_coeffs = np.poly(eigenvalues)
    return polynomial_coeffs

def nilai_eigen():
    # Input matriks dari pengguna
    print("=========================================")
    print("              EIGEN VALUES               ")
    print("=========================================")
    n = int(input("Masukkan ukuran matriks (n x n): "))
    matriks = []
    print("Masukkan elemen-elemen matriks:")
    for i in range(n):
        row = list(map(float, input().split()))
        matriks.append(row)
    matrix1 = str(matriks)
    # Hitung nilai eigen, vektor eigen, dan persamaan polinomial
    eigenvalues, eigenvectors = hitung_eigen(matriks)
    print("Nilai eigen dari matriks tersebut adalah:", eigenvalues)
    print("Vektor eigen dari matriks tersebut adalah:")
    for i in range(len(eigenvalues)):
        print("Nilai eigen:", eigenvalues[i])
        print("Vektor eigen:")
        print(eigenvectors[:, i])

    polynomial_coeffs = hitung_persamaan_polinomial(matriks)
    result = np.poly1d(polynomial_coeffs)
    poly_string = " + ".join([f"{result.coeffs[i]} * x^{len(result.coeffs)-1-i}" for i in range(len(result.coeffs))])
    print("Persamaan polinomial dari nilai eigen adalah:\n", poly_string)

    # Cek apakah matriks invers dari eigenvectors ada
    try:
        eigenvectors_inv = np.linalg.inv(eigenvectors)
        print("Matriks P invers ditemukan.")
        print("Matriks P:")
        print(eigenvectors)
        print("Matriks P invers:")
        print(eigenvectors_inv)
    except np.linalg.LinAlgError:
        print("Tidak dapat menemukan invers dari matriks P.")

    with open('hasil.txt', 'a') as file:
        file.write("Soal Eigen:\n")
        file.write(matrix1 + "\n\n")
        file.write("Output :\n")
        file.write("eigenvalue dan eigenvector = \n")
        for i in range(len(eigenvalues)):
            file.write("Nilai eigen: " + str(eigenvalues[i]) + "\n")
            file.write("Vektor eigen: \n" + str(eigenvectors[:, i]) + "\n")
        file.write("polynomial = \n")
        file.write(poly_string + "\n\n")
        if 'eigenvectors_inv' in locals():
            file.write("Matriks P:\n")
            file.write(str(eigenvectors) + "\n\n")
            file.write("Matriks P invers:\n")
            file.write(str(eigenvectors_inv) + "\n\n")

# =============================================== DIAGONAL MATRIX SECTION ===================================================

def matriks_diagonal():
    print("=========================================")
    print("             DIAGONAL MATRIX             ")
    print("=========================================")
    n = int(input("Masukkan jumlah baris: "))
    m = int(input("Masukkan jumlah kolom: "))

    print(f"Masukkan koefisien matriks A (ukuran {n}x{m}):")
    A = []
    for _ in range(n):
        row = [float(x) for x in input().split()]
        A.append(row)
    A = np.array(A)


    eigenvalues, eigenvectors = np.linalg.eig(A)
    p = eigenvectors
    d = np.diag(eigenvalues)
    d = np.round(d,4)
    pap = np.linalg.inv(p) @ A @ p
    pap = np.round(pap, 4)
    p = np.round(p,4)

    result = (f"Soal Diagonal Matriks:\nMatriks A =\n{A}\nOutput :\nMatriks D = \n{d}\nMatriks P = \n{p}\nMatriks P^-1AP =\n{pap}\n")
    print(result)
    with open('hasil.txt', 'a') as file:
        file.write(result + "\n")

# =============================================== INVERSE MATRIX SECTION ===================================================

def matriks_invers():
    print("=========================================")
    print("             MATRIX INVERSE              ")
    print("=========================================")
    n = int(input("Masukkan jumlah baris: "))
    m = int(input("Masukkan jumlah kolom: "))
    print(f"Masukkan koefisien matriks A (ukuran {n}x{m}):")
    A = []
    for _ in range(n):
        row = [float(x) for x in input().split()]
        A.append(row)
    A = np.array(A)

    inverse_matriks = np.linalg.inv(A)

    result = (f"Soal Inverse Matriks:\nMatriks A =\n{A}\nOutput :\nMatriks A^-1=\n{inverse_matriks}\n")
    print(result)
    with open('hasil.txt', 'a') as file:
        file.write(result + "\n")

# =============================================== DETERMINANT SECTION ===================================================

def eliminasi_gauss(matriks):
    det = 1
    n = len(matriks)
    for i in range(n):
        # Pivoting
        max_index = i
        for j in range(i + 1, n):
            if abs(matriks[j][i]) > abs(matriks[max_index][i]):
                max_index = j
        matriks[i], matriks[max_index] = matriks[max_index], matriks[i]

        # Pengurangan baris
        for j in range(i + 1, n):
            ratio = matriks[j][i] / matriks[i][i]
            for k in range(i, n):
                matriks[j][k] -= ratio * matriks[i][k]

    # Hitung determinan
    for i in range(n):
        det *= matriks[i][i]
    return det

def eliminasi_gauss_jordan(matriks):
    n = len(matriks)
    for i in range(n):
        # Pivoting
        max_index = i
        for j in range(i + 1, n):
            if abs(matriks[j][i]) > abs(matriks[max_index][i]):
                max_index = j
        matriks[i], matriks[max_index] = matriks[max_index], matriks[i]

        # Normalisasi baris
        divisor = matriks[i][i]
        for k in range(i, n):
            matriks[i][k] /= divisor

        # Pengurangan baris
        for j in range(n):
            if j != i:
                ratio = matriks[j][i]
                for k in range(i, n):
                    matriks[j][k] -= ratio * matriks[i][k]

    # Hitung determinan
    det = 1
    for i in range(n):
        det *= matriks[i][i]
    return det

def matriks_balikan(matriks):
    det = np.linalg.det(matriks)
    if det == 0:
        return "Determinan matriks nol, tidak bisa menghitung matriks balikan."
    else:
        matriks_balik = np.linalg.inv(matriks)
        return matriks_balik

def hitung_determinan(matriks, metode):
    if metode == "1":
        return eliminasi_gauss(matriks)
    elif metode == "2":
        return eliminasi_gauss_jordan(matriks)
    elif metode == "3":
        return matriks_balikan(matriks)
    else:
        return "Metode tidak valid"

def determinan():
    print("=========================================")
    print("               DETERMINAN                ")
    print("=========================================")
    print("1. Metode eliminasi Gauss")
    print("2. Metode eliminasi Gauss-Jordan")
    print("3. Metode matriks balikan")
    metode = input("Pilihan Anda: ")

    # Input matriks dari pengguna
    n = int(input("Masukkan ukuran matriks (n x n): "))
    matriks = []
    print("Masukkan elemen-elemen matriks:")
    for i in range(n):
        row = list(map(float, input().split()))
        matriks.append(row)
    matrix1 = str(matriks)
    # Hitung determinan
    determinan = hitung_determinan(matriks, metode)
    determinan = str(determinan)
    print("Determinan matriks tersebut adalah:", determinan)
    with open('hasil.txt', 'a') as file:
        file.write("Soal Determinan:\n")
        file.write(matrix1 + "\n\n")
        file.write("Output :\n")
        file.write("Determinan matriks =\n")
        file.write(determinan)
        file.write("\n\n")

# =============================================== MAIN FUNCTION SECTION ===================================================

def main():
    while True:
        os.system('cls' if os.name == 'nt' else 'clear')
        menu()
        selection = int(input("Pilihan Anda: "))

        match selection: 
            case 1:
                spl()
                input("\nPress any key to continue . . . ")
            case 2:
                determinan()
                input("\nPress any key to continue . . . ")
            case 3:
                matriks_invers()
                input("\nPress any key to continue . . . ")
            case 4:
                LU()
                input("\nPress any key to continue . . . ")
            case 5:
                nilai_eigen()
                input("\nPress any key to continue . . . ")
            case 6:
                matriks_diagonal()
                input("\nPress any key to continue . . . ")
            case 7:
                svd()
                input("\nPress any key to continue . . . ")
            case 8:
                print("\nTerima kasih telah menggunakan program kalkulator matriks!")
                break
            case _:
                print("Pilihan tidak valid")

if __name__ == "__main__":
    main()


