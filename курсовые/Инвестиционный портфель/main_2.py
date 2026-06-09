import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Настройка визуализации
sns.set(style="whitegrid")


# 1. ЗАГРУЗКА И ПРЕДОБРАБОТКА ДАННЫХ

try:
    # Загружаем данные. Убедись, что имя файла совпадает с тем, что лежит в папке
    df = pd.read_csv('BMW_stock_data.csv', parse_dates=['Date'], index_col='Date')
    print("Данные акций BMW успешно загружены!")
    print(f"Период данных: с {df.index.min().strftime('%Y-%m-%d')} по {df.index.max().strftime('%Y-%m-%d')}\n")
except FileNotFoundError:
    print("Ошибка: Файл с ценами акций не найден. Проверьте путь и имя файла (BMW.csv).")
    exit()

# Для анализа инвесторы всегда используют 'Adj_Close' (скорректированная цена закрытия)
# Создаем отдельный датафрейм только с этой ценой
bmw_prices = df[['Adj_Close']].copy()

# Вычисляем ежедневную доходность
bmw_prices['Daily_Return'] = bmw_prices['Adj_Close'].pct_change()

# Удаляем первую строчку, так как для неё нет предыдущего дня (там будет NaN)
bmw_prices = bmw_prices.dropna()


# 2. АНАЛИЗ РИСКА И ДОХОДНОСТИ ПО ГОДАМ

print("--- Расчёт финансовых метрик по годам ---")

# Группируем данные по годам
bmw_prices['Year'] = bmw_prices.index.year

# Считаем годовую доходность и годовой риск (волатильность)
# В году примерно 252 рабочих торговых дня
yearly_stats = bmw_prices.groupby('Year')['Daily_Return'].agg(
    Annual_Return=lambda x: x.mean() * 252,
    Annual_Volatility=lambda x: x.std() * np.sqrt(252)
).reset_index()

# Считаем Коэффициент Шарпа для каждого года (Доходность / Риск)
yearly_stats['Sharpe_Ratio'] = yearly_stats['Annual_Return'] / yearly_stats['Annual_Volatility']

print(yearly_stats.to_string(index=False))


# 3. ПОИСК ЛУЧШЕГО И ХУДШЕГО ГОДА

best_year_row = yearly_stats.loc[yearly_stats['Sharpe_Ratio'].idxmax()]
worst_year_row = yearly_stats.loc[yearly_stats['Sharpe_Ratio'].idxmin()]

print("\n=== РЕЗУЛЬТАТЫ АНАЛИЗА ДЛЯ КУРСОВОЙ ===")
print(f" Лучший год для инвесторов BMW: {int(best_year_row['Year'])} г.")
print(f"  • Годовая доходность: {best_year_row['Annual_Return']*100:.2f}%")
print(f"  • Волатильность (Риск): {best_year_row['Annual_Volatility']*100:.2f}%")
print(f"  • Коэффициент Шарпа: {best_year_row['Sharpe_Ratio']:.2f}")

print(f"\n Худший год для инвесторов BMW: {int(worst_year_row['Year'])} г.")
print(f"  • Годовая доходность: {worst_year_row['Annual_Return']*100:.2f}%")
print(f"  • Волатильность (Риск): {worst_year_row['Annual_Volatility']*100:.2f}%")
print(f"  • Коэффициент Шарпа: {worst_year_row['Sharpe_Ratio']:.2f}")


# 4. ВИЗУАЛИЗАЦИЯ

plt.figure(figsize=(12, 6))

# Строим график зависимости Доходности от Риска по годам
sc = plt.scatter(yearly_stats['Annual_Volatility'], yearly_stats['Annual_Return'], 
                 c=yearly_stats['Sharpe_Ratio'], cmap='coolwarm', s=300, edgecolors='black', alpha=0.8)

plt.colorbar(sc, label='Коэффициент Шарпа (Эффективность инвестиций)')

# Подписываем года на графике
for i, txt in enumerate(yearly_stats['Year']):
    plt.annotate(int(txt), (yearly_stats['Annual_Volatility'].iloc[i], yearly_stats['Annual_Return'].iloc[i]),
                 textcoords="offset points", xytext=(0,10), ha='center', fontsize=9)

plt.title('Анализ риска и доходности акций BMW по годам (1996-2026)')
plt.xlabel('Годовой Риск (Волатильность)')
plt.ylabel('Годовая Ожидаемая Доходность')
plt.axhline(0, color='red', linestyle='--', linewidth=1, alpha=0.5) # Линия нулевой доходности
plt.tight_layout()
plt.show()