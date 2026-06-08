import numpy as np
import matplotlib.pyplot as plt

def linear_congruential(x0, a, c, M, k):
    """ 
    x0 is the initial number, 
    a is the multiplyer
    c is the constant added
    M is the modulus
    """

    x_list = [x0]
    current_x = x0

    for i in range (k):
        current_x = (a * current_x + c) % M
        x_list.append(current_x)
    
    return np.array(x_list)/M


def show_hist(U, bins=100 ):
    plt.hist(U, bins=bins)
    plt.xlabel('Value')
    plt.ylabel('count')
    plt.title('Distributions of random generated values')
    plt.show()
    return

def show_scatter(U):
    plt.scatter(U[:-1], U[1:], s=1)
    plt.title('Scatter of concecutive points agains each other')    


def perform_statistical_test_X2(U,critical_value,m=0 , classes = 10):
    print()
    expected_n = len(U)/classes
    observed_n, bin_edges = np.histogram(U, bins=classes)

    T = np.sum((observed_n-expected_n)**2/expected_n)

    df = classes-1-m
    #print(f'Degrees of fredom is {df} which corresponds to {critical_value} (look up)')
    if T<critical_value:
        print(f"PASSED : (statistical_test_X2) T value T = {T} is less than critical value {critical_value}")
    else: 
        print(f"FAILED : (statistical_test_X2) T value T = {T} is higher than critical value {critical_value}")
    return T


def perform_Kolmogorov_Smirnov_test(U, classes, critical_value):
    print()
    expected_n = len(U)/classes
    observed_n, bin_edges = np.histogram(U, bins=classes)
    expected_list = [i*expected_n for i in range(classes)]
    observed_n = np.array(observed_n)
    observed_list = [int(np.sum(observed_n[:i])) for i in range(len(observed_n))]
    observed_list, expected_list
    n = len(U)
    observed_list =np.array(observed_list)/n
    expected_list = np.array(expected_list)/n
    plt.plot(expected_list, linewidth=8)

    plt.plot(list(range(0, classes)), observed_list, c='r', marker='o')

    Dn = np.max(np.abs(observed_list-expected_list))
    if Dn<critical_value:
        print(f"PASSED : (Kolmogorov_Smirnov_test) Dn value Dn = {Dn} is less than critical value {critical_value}")
    else: 
        print(f"FAILED : (Kolmogorov_Smirnov_test) Dn value Dn = {Dn} is higher than critical value {critical_value}")
    return Dn



def perform_run_test1(U,critical_value):
    print()
    # Run test 1
    median_ = np.median(U)
    # n1: number of samples above median
    n1 = np.sum(U>median_)
    # n2: number of samples below median
    n2 = np.sum(U<median_)

    Ra = 0
    Rb = 0
    runs_above = U>median_
    runs_below = U<median_

    if runs_above[0]==True:
        Ra+=1
    if runs_below[0]==True:
        Rb+=1
    for i in range(1, len(U)):

        # mark every start
        if runs_above[i]==True and runs_above[i-1]==False:
            Ra +=1
        if runs_below[i]==True and runs_below[i-1]==False:
            Rb +=1

    T = Ra + Rb
    #print(f"Ra = {Ra} and Rb = {Rb} \nT={T}")

    # gausian (normal dist)
    mean_normal = 2*n1*n2/(n1+n2)+1
    variance_normal = 2* n1*n2*(2*n1*n2-n1-n2)/((n1+n2)**2*(n1+n2-1))

    z_score = (T-mean_normal)/np.sqrt(variance_normal)
    #print(f"\nz-score = {z_score}")
    if z_score<critical_value:
        print(f"PASSED : (Run-test1) z_score value z_score = {z_score} is less than critical value {critical_value}")
    else: 
        print(f"FAILED : (Run-test1) z_score value z_score = {z_score} is higher than critical value {critical_value}")
    return z_score


def perform_run_test2(U,critical_value):
    print()
    n = len(U)
    # Run test II ()Up/down from Knuth
    r =1

    index_shifts = [0]
    for i in range(1, len(U)):
        # mark every start
        if U[i-1] > U[i] :
            r +=1
            index_shifts.append(i)
    if i not in index_shifts:
        index_shifts.append(i)


    #print(f"Runs r = {r} \n")

    Run_lengths = np.diff(index_shifts)

    R = np.zeros(6).astype(int)

    R[0] = int(np.sum(Run_lengths==1))
    R[1] = int(np.sum(Run_lengths==2))
    R[2] = int(np.sum(Run_lengths==3))
    R[3] = int(np.sum(Run_lengths==4))
    R[4] = int(np.sum(Run_lengths==5))
    R[5] = int(np.sum(Run_lengths>5))
    A = np.array([
        [4529.4,  9044.9,  13568.0,  18091.0,  22615.0,  27892.0],
        [9044.9, 18097.0,  27139.0,  36187.0,  45234.0,  55789.0],
        [13568.0, 27139.0,  40721.0,  54281.0,  67852.0,  83685.0],
        [18091.0, 36187.0,  54281.0,  72414.0,  90470.0, 111580.0],
        [22615.0, 45234.0,  67852.0,  90470.0, 113262.0, 139476.0],
        [27892.0, 55789.0,  83685.0, 111580.0, 139476.0, 172860.0]
        ])

    B = np.array([
        1/6,
        5/24,
        11/120,
        19/720,
        29/5040,
        1/840
    ])
    #print(f"R = {R}")

    Z = (1 / (n - 6)) * (R - n * B).T @ A @ (R - n * B)
    #print(f"Z = {Z}")
    if Z<critical_value:
        print(f"PASSED :(Run-test2) Z value Z = {Z} is less than critical value {critical_value}")
    else: 
        print(f"FAILED : (Run-test2) Z value Z = {Z} is higher than critical value {critical_value}")
    return Z


def perform_correlation_test(U,critical_value,h=2):
    # h is the lag we want to skip with 
    print()


    indexes_to_search = np.arange(0,len(U),h)
    n = len(U)

    UiUih = [U[i]*U[i+h] for i in range(n-h)]
    UiUih = np.array(UiUih)
    ch = 1/(n-h)* np.sum(UiUih)
    #print(f"ch= {ch}")
    expected_mean = 0.25
    variance = 7 / (144 * n)     # Notice the parentheses around (144 * k)
    std_dev = np.sqrt(variance)

    # 3. Calculate the actual Z-test statistic using your calculated 'c'
    Z = (ch - expected_mean) / std_dev

    if Z<critical_value:
        print(f"PASSED : (Correlation test) Z value Z = {Z} is less than critical value {critical_value}")
    else: 
        print(f"FAILED : (Correlation test) Z value Z = {Z} is higher than critical value {critical_value}")
    return Z
