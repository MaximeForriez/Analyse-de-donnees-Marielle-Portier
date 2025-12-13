import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

def calculate_stats(distribution_name, params):
    """
    Calcule la moyenne et l'écart type pour une distribution donnée
    """
    try:
        if distribution_name == "dirac":
            # Pour la loi de Dirac, la moyenne est la position du pic et l'écart type est 0
            return params["position"], 0
        
        elif distribution_name == "uniform_discrete":
            n = params["n"]
            # Pour la loi uniforme discrète
            mean = (n - 1) / 2
            var = (n**2 - 1) / 12
            return mean, np.sqrt(var)
        
        elif distribution_name == "binomial":
            n, p = params["n"], params["p"]
            mean = n * p
            std = np.sqrt(n * p * (1 - p))
            return mean, std
        
        elif distribution_name == "poisson":
            lambda_param = params["lambda"]
            # Pour la loi de Poisson, moyenne = variance = lambda
            return lambda_param, np.sqrt(lambda_param)
        
        elif distribution_name == "zipf":
            a = params["a"]
            # Pour la loi de Zipf-Mandelbrot
            return stats.zipf.stats(a, moments='mv')
        
        elif distribution_name == "normal":
            mu, sigma = params["mu"], params["sigma"]
            return mu, sigma
        
        elif distribution_name == "lognormal":
            s = params["s"]
            # Pour la loi log-normale
            mean = np.exp(s**2 / 2)
            var = (np.exp(s**2) - 1) * np.exp(s**2)
            return mean, np.sqrt(var)
        
        elif distribution_name == "uniform_continuous":
            a, b = params["a"], params["b"]
            mean = (a + b) / 2
            std = np.sqrt((b - a)**2 / 12)
            return mean, std
        
        elif distribution_name == "chi2":
            df = params["df"]
            mean = df
            std = np.sqrt(2 * df)
            return mean, std
        
        elif distribution_name == "pareto":
            b = params["b"]
            if b > 1:
                mean = b / (b - 1)
                if b > 2:
                    var = (b * mean**2) / ((b - 1)**2 * (b - 2))
                    return mean, np.sqrt(var)
                return mean, float('inf')
            return float('inf'), float('inf')
            
    except Exception as e:
        return f"Erreur dans le calcul : {str(e)}", None

def print_distribution_stats(name, params):
    """
    Affiche les statistiques d'une distribution
    """
    mean, std = calculate_stats(name, params)
    if isinstance(mean, (int, float)) and isinstance(std, (int, float)):
        print(f"{name}:")
        print(f"  Moyenne: {mean:.4f}")
        print(f"  Écart-type: {std:.4f}")
        print()
    else:
        print(f"{name}: {mean}")
        print()

def plot_discrete_distributions():
    # Création de la figure avec sous-graphiques
    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Distributions de variables discrètes')

    # Loi de Dirac
    x_dirac = np.arange(-2, 3)
    dirac = np.zeros_like(x_dirac)
    dirac[2] = 1  # Pic à x=0
    ax1.stem(x_dirac, dirac)
    ax1.set_title('Loi de Dirac')

    # Loi uniforme discrète
    n_uniform = 10
    x_uniform = np.arange(n_uniform)
    uniform = np.full(n_uniform, 1/n_uniform)
    ax2.stem(x_uniform, uniform)
    ax2.set_title('Loi uniforme discrète')

    # Loi binomiale
    n, p = 20, 0.5
    x_binom = np.arange(0, n+1)
    binom = stats.binom.pmf(x_binom, n, p)
    ax3.stem(x_binom, binom)
    ax3.set_title(f'Loi binomiale (n={n}, p={p})')

    # Loi de Poisson (discrète)
    lambda_poisson = 4
    x_poisson = np.arange(0, 15)
    poisson = stats.poisson.pmf(x_poisson, lambda_poisson)
    ax4.stem(x_poisson, poisson)
    ax4.set_title(f'Loi de Poisson (λ={lambda_poisson})')

    # Loi de Zipf-Mandelbrot
    a = 2.0  # paramètre de la loi
    x_zipf = np.arange(1, 21)
    zipf = stats.zipf.pmf(x_zipf, a)
    ax5.stem(x_zipf, zipf)
    ax5.set_title(f'Loi de Zipf-Mandelbrot (a={a})')

    # Ajuster la mise en page
    ax6.axis('off')
    plt.tight_layout()
    plt.show()

def plot_continuous_distributions():
    # Création de la figure avec sous-graphiques
    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Distributions de variables continues')

    # Loi de Poisson (version continue approchée)
    x_poisson = np.linspace(0, 15, 1000)
    lambda_poisson = 4
    poisson = stats.gamma.pdf(x_poisson, a=lambda_poisson, scale=1)
    ax1.plot(x_poisson, poisson)
    ax1.set_title(f'Loi de Poisson continue (λ={lambda_poisson})')

    # Loi normale
    mu, sigma = 0, 1
    x_normal = np.linspace(mu - 4*sigma, mu + 4*sigma, 1000)
    normal = stats.norm.pdf(x_normal, mu, sigma)
    ax2.plot(x_normal, normal)
    ax2.set_title(f'Loi normale (μ={mu}, σ={sigma})')

    # Loi log-normale
    x_lognorm = np.linspace(0, 5, 1000)
    s = 0.5
    lognorm = stats.lognorm.pdf(x_lognorm, s)
    ax3.plot(x_lognorm, lognorm)
    ax3.set_title(f'Loi log-normale (s={s})')

    # Loi uniforme continue
    a, b = 0, 1
    x_uniform = np.linspace(-0.5, 1.5, 1000)
    uniform = stats.uniform.pdf(x_uniform, a, b-a)
    ax4.plot(x_uniform, uniform)
    ax4.set_title(f'Loi uniforme ({a}, {b})')

    # Loi du chi-2
    df = 3  # degrés de liberté
    x_chi2 = np.linspace(0, 10, 1000)
    chi2 = stats.chi2.pdf(x_chi2, df)
    ax5.plot(x_chi2, chi2)
    ax5.set_title(f'Loi du χ² (df={df})')

    # Loi de Pareto
    b = 2.5  # paramètre de forme
    x_pareto = np.linspace(1, 5, 1000)
    pareto = stats.pareto.pdf(x_pareto, b)
    ax6.plot(x_pareto, pareto)
    ax6.set_title(f'Loi de Pareto (b={b})')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Afficher les statistiques pour les distributions discrètes
    print("=== Statistiques des distributions discrètes ===")
    print_distribution_stats("dirac", {"position": 0})
    print_distribution_stats("uniform_discrete", {"n": 10})
    print_distribution_stats("binomial", {"n": 20, "p": 0.5})
    print_distribution_stats("poisson", {"lambda": 4})
    print_distribution_stats("zipf", {"a": 2.0})

    print("=== Statistiques des distributions continues ===")
    print_distribution_stats("normal", {"mu": 0, "sigma": 1})
    print_distribution_stats("lognormal", {"s": 0.5})
    print_distribution_stats("uniform_continuous", {"a": 0, "b": 1})
    print_distribution_stats("chi2", {"df": 3})
    print_distribution_stats("pareto", {"b": 2.5})

    # Générer et sauvegarder les distributions discrètes
    plot_discrete_distributions()
    plt.savefig('distributions_discretes.png')
    plt.close()

    # Générer et sauvegarder les distributions continues
    plot_continuous_distributions()
    plt.savefig('distributions_continues.png')
    plt.close()