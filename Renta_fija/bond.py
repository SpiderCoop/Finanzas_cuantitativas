"""
Descripción:   Clase para valuacion de bonos de distintos tipos
Autor:         David Jiménez Cooper - SpiderCoop
Fecha:         2026-06-14
"""

import pandas as pd
import numpy as np
import math


# Definimos la clase bono
class Bond:
    """
    Clase para modelar y valuar un bono con tasa fija o variable, 
    incluyendo cálculo de precios, cupon_flows, intereses devengados y sensibilidades.

    Parámetros
    ----------
    cupon_period : int
        Número de días que corresponden a un período de cupón.
    spread : float
        spread a añadir a la tasa de cupón si corresponde (expresada en porcentaje).
    emition_date : pd.Timestamp
        Fecha de emisión del bono. Si no se proporciona, se asume None.
    expire_date : pd.Timestamp
        Fecha de vencimiento del bono.

    calendar_convention : str, opcional
        Convención de calendario a utilizar para el cómputo de fracciones de año.
        Opciones: 'actual/actual', 'actual/360', 'actual/365', '30/360', '30/365'.
        Por defecto 'Actual/360'.

    Atributos
    ---------
    precio_sucio : float
        Precio total del bono, incluyendo intereses devengados.
    precio_limpio : float
        Precio del bono sin considerar intereses devengados.
    intereses_devengados : float
        Intereses acumulados desde el último pago de cupón.
    delta : float
        Sensibilidad de primer orden del precio respecto a cambios en la tasa de rendimiento.
    gamma : float
        Sensibilidad de segundo orden (convexidad) del precio respecto a cambios en la tasa de rendimiento.
    """
    def __init__(self, cupon_period:int, spread:float, emition_date:pd.Timestamp, expire_date:pd.Timestamp, amortizing:bool|str = False, compounding_method:str = 'compound', calendar_convention:str = 'Actual/Actual'):
        
        # Convertimos los parametros a tipo correspondiente
        self.cupon_period = int(cupon_period)
        self.spread = self._ensure_decimal_rate(spread)

        # Convertimos las fechas a tipo datetime
        self.expire_date = pd.to_datetime(expire_date)
        self.emition_date = pd.to_datetime(emition_date)

        # Variables para amortizacion
        if isinstance(amortizing,str):
            self.amortizing = amortizing.strip().lower()
            assert self.amortizing in ['lineal', 'french'], 'Amortizing must be lineal or french'
        else:
            self.amortizing = amortizing

        # Variable indicadora de la metodología para descuento simple o compuesto
        self.compounding_method = compounding_method.strip().lower()
        assert self.compounding_method in ['simple', 'compound'], 'Compounding method must be simple or compound'
        
        # Convencion de calendario
        self.calendar_convention = calendar_convention.strip().lower()


    def day_count_fraction(self,start_date: pd.Timestamp, end_date: pd.Timestamp):
        """
        Calcula el número de días y la fracción de año entre dos fechas
        según la convención de calendario especificada en el bono.

        Parámetros
        ----------
        start_date : pd.Timestamp
            Fecha inicial.
        end_date : pd.Timestamp
            Fecha final.

        Retorna
        -------
        days : int
            Número de días entre las fechas.
        fraction : float
            Fracción del año transcurrido según la convención.
        """
        
        # Validación
        if not isinstance(start_date, pd.Timestamp) or not isinstance(end_date, pd.Timestamp):
            raise TypeError("Las fechas deben ser de tipo pd.Timestamp")

        assert start_date < end_date, 'end date must be after star date'

        # Días reales entre fechas
        days = (end_date - start_date).days

        # Actual/Actual ISDA
        if self.calendar_convention == "actual/actual":

            # Calculamos la fraccion del año por segmento del año tomando en cuenta años biciestos
            fraction = 0.0
            current_date = start_date
            while current_date < end_date:
                year_end = pd.Timestamp(year=current_date.year, month=12, day=31)
                period_end = min(year_end, end_date)

                days_in_period = (period_end - current_date).days
                
                # Ajuste para contabilizar el último dia del año dado que no se contabiliza al actualizar current day, pero en la convencion ISDA no se contabiliza el dia de vencimiento
                if period_end != end_date:
                    days_in_period +=1

                days_in_year = 366 if current_date.is_leap_year else 365

                fraction += days_in_period / days_in_year

                current_date = period_end + pd.Timedelta(days=1)

            return days, fraction

        # --- Otras convenciones ---
        elif self.calendar_convention == "actual/360":
            fraction = days / 360

        elif self.calendar_convention == "actual/365":
            fraction = days / 365

        elif self.calendar_convention == "30/360":
            d1 = min(start_date.day, 30)
            d2 = min(end_date.day, 30) if start_date.day == 30 else end_date.day
            days = (end_date.year - start_date.year) * 360 + \
                    (end_date.month - start_date.month) * 30 + \
                    (d2 - d1)
            fraction = days / 360

        elif self.calendar_convention == "30/365":
            d1 = min(start_date.day, 30)
            d2 = min(end_date.day, 30) if start_date.day == 30 else end_date.day
            days = (end_date.year - start_date.year) * 365 + \
                    (end_date.month - start_date.month) * 30 + \
                    (d2 - d1)
            fraction = days / 365

        else:
            raise ValueError(f"Convención '{self.calendar_convention}' no reconocida")

        return days, fraction
    
    @staticmethod
    def _ensure_decimal_rate(interest_rate):
        """
        Convierte tasas a decimales. Soporta float, list, numpy array o pandas Series.
        """
        if isinstance(interest_rate, pd.Series):
            return interest_rate.apply(lambda x: x/100 if x > 1 else x)
        elif isinstance(interest_rate, (list, np.ndarray)):
            return pd.Series(interest_rate).apply(lambda x: x/100 if x > 1 else x).values
        elif isinstance(interest_rate, (int, float)):
            return interest_rate/100 if interest_rate > 1 else interest_rate
        else:
            raise TypeError("Rate must be float, int, list, numpy array, or pandas Series")
        


    def _structure_capital_payments(self):
        
        # Get the cupon interest_rate 
        cupon_rate = self.cupon_rate[self.days_to_cupon_payments] if isinstance(self.cupon_rate, pd.Series) else self.cupon_period

        if self.amortizing:
            if self.amortizing == 'lineal':
                capital_payment = self.face_value / self.pending_cupons

            elif self.amortizing == 'french':
                capital_payment = self.face_value * cupon_rate / (1 - (1 + cupon_rate) ** -self.pending_cupons)

            else:
                raise ValueError('Method not acknowledge')
            
            if isinstance(capital_payment, (np.ndarray, pd.Series)):
                capital_payments = capital_payment

            else:
                capital_payments = pd.Series(capital_payment, index=self.days_to_cupon_payments)

            face_values = self.face_value - capital_payments
            face_values.shift().bfill()

        else:
            capital_payments = pd.Series(0, index=self.days_to_cupon_payments)
            capital_payments.iloc[-1] = self.face_value
            face_values = pd.Series(self.face_value, index=self.days_to_cupon_payments)

        # Change the name to align the different pd.series
        capital_payments.index.name = 'Plazo'
        face_values.index.name = 'Plazo'

        return capital_payments, face_values



    # Estruturar los cupon_flows de acuerdo con el tipo de datos y tipo de bono que se proporcione
    def _structure_cashflows(self):
        """
        Estructura los cupon_flows de efectivo del bono según su tipo (cupón fijo, variable o con spread).

        Parámetros
        ----------
        days_in_year : int | np.ndarray
            Número de días del año usados para calcular los cupon_flows según convención de calendario.

        Retorna
        -------
        cupon_flows : pd.Series
            Serie con los cupon_flows de efectivo (cupones + amortización de principal).
        interest_rate : float | pd.Series
            Tasa de rendimiento correspondiente a los cupon_flows.
        """

        # Compute the dates of payment
        self.dates_cupon_payments = np.array([self.valuation_date + pd.Timedelta(days=num_day) for num_day in self.days_to_cupon_payments])  # fechas de pagos de los siguientes cupones

        # Get the number of days in the year according to the date of payment for each date
        if self.calendar_convention == 'actual/actual':
            days_in_year = np.array([366 if date.is_leap_year else 365 for date in self.dates_cupon_payments])
        elif self.calendar_convention.split('/')[-1] == "360":
            days_in_year = np.array([360 for _ in self.dates_cupon_payments])
        elif self.calendar_convention.split('/')[-1] == "365":
            days_in_year = np.array([365 for _ in self.dates_cupon_payments])

        # Compute the capital payments and the face values according to the amortization structure
        capital_payments, face_values = self._structure_capital_payments()

        # Get the cupon interest_rate or rates and the corresponding cupon value
        cupon_rate = self.cupon_rate[self.days_to_cupon_payments] if isinstance(self.cupon_rate, pd.Series) else self.cupon_rate
        cupon_flows = face_values * cupon_rate * self.cupon_period / days_in_year

        if isinstance(cupon_rate, pd.Series) or (isinstance(cupon_rate, float) and cupon_rate > 0):
            self._cupon_value_t0 = cupon_flows.iloc[0]
        else:
            self._cupon_value_t0 = face_values.iloc[-1]

        if self.spread != 0 and isinstance(cupon_rate, float):
            cupon_rate_adj = cupon_rate + self.spread
            rate_adj = self.interest_rate + self.spread
            cupon_value_adj= face_values * cupon_rate_adj * self.cupon_period / days_in_year  # Valor del cupon ajustado por spread
            cupon_flows.iloc[1:] = cupon_value_adj.iloc[1:]
        
        else:
            rate_adj = self.interest_rate

        # Add the capital payments to the cupon payments
        total_flows = cupon_flows + capital_payments
    
        return total_flows, rate_adj, days_in_year



    # Caluclo de precio sucio, precio limpio e intereses devengados
    def valuate(self, face_value:float, interest_rate:float|pd.Series, cupon_rate:float|pd.Series, valuation_date:pd.Timestamp = pd.Timestamp.today()):
        """
        Valúa el bono en una fecha determinada calculando el precio sucio, 
        limpio e intereses devengados, así como las sensibilidades.

        Parámetros
        ----------
        face_value : float
            Valor nominal del bono.
        interest_rate : float | pd.Series
            Tasa de rendimiento de mercado (en decimal si es float, o curva de tasas si es pd.Series).
        cupon_rate : float | pd.Series
            Tasa de cupón del bono (en decimal si es float, o vector de tasas si es pd.Series).
        valuation_date : pd.Timestamp, opcional
            Fecha de valuación del bono. Por defecto, la fecha actual.

        Atributos calculados
        --------------------
        precio_sucio : float
            Precio total del bono (incluyendo intereses devengados).
        precio_limpio : float
            Precio del bono sin considerar intereses devengados.
        intereses_devengados : float
            Intereses acumulados desde el último pago de cupón.
        discount_factor : np.ndarray
            Factores de descuento aplicados a cada flujo.
        valor_presente_flujos : pd.Series
            Valor presente de los cupon_flows futuros.
        """

        self.face_value = float(face_value)
        self.interest_rate = self._ensure_decimal_rate(interest_rate)
        self.cupon_rate = self._ensure_decimal_rate(cupon_rate)

        # Convertimos a tipo datetime
        self.valuation_date = pd.to_datetime(valuation_date)
        
        # Compute number of pending cupons
        self.days_to_expire, _ = self.day_count_fraction(self.valuation_date, self.expire_date)
        pending_cupons_flt = self.days_to_expire / self.cupon_period  # Numero de cupones pendientes por pagar (con fraccion de cupon debido a los dias devengados)
        self.pending_cupons = math.ceil(pending_cupons_flt)                 # Numero real de cupones pendientes por pagar

        # Compute array of number of days to the next pending cupons and their dates
        self.accrued_days = (self.pending_cupons - pending_cupons_flt) * self.cupon_period  # Dias devengados desde el ultimo pago de cupon
        days_to_next_cupon = math.ceil(self.cupon_period - self.accrued_days)  # Dias al vencimiento del proximo cupon
        self.days_to_cupon_payments = days_to_next_cupon + self.cupon_period * np.arange(self.pending_cupons)   # Numero de dias para los siguientes pagos de cupon


        # Estruturar los cupon_flows
        self.total_flows, interest_rate, days_in_year = self._structure_cashflows()

        # Obtenemos las tasas de rendimiento correspondientes
        if isinstance(interest_rate, pd.Series):
            interest_rate = interest_rate[self.days_to_cupon_payments]

        # Calculamos el factor de descuento dependiendo de si es bono cuponado o cupon cero
        if self.compounding_method == 'compound':
            self.discount_factor = (1+interest_rate*self.cupon_period/days_in_year)**(-self.days_to_cupon_payments/self.cupon_period)
        elif self.compounding_method == 'simple':
            self.discount_factor = 1/(1+interest_rate*self.days_to_cupon_payments/days_in_year)

        # Calculamos los valores presentes de los cupon_flows (los guardamos dado que ayudan a calcular las sensibilidades)
        self.pv_total_flows = self.total_flows*self.discount_factor

        # Calculo de precios
        self.precio_sucio = self.pv_total_flows.sum()
        self.accrued_interests = self._cupon_value_t0*self.accrued_days/self.cupon_period
        self.precio_limpio = self.precio_sucio-self.accrued_interests

        # Calculammos las sensibilidades de acuerdo a la nueva valuar
        self.delta, self.gamma = self.compute_sensibilities()

        return self.precio_sucio, self.accrued_interests, self.precio_limpio


    # Calculo de sensibildades
    def compute_sensibilities(self):
        """
        Calcula las sensibilidades del bono (delta y gamma) 
        a partir del valor presente de los cupon_flows.

        Retorna
        -------
        delta : float
            Sensibilidad de primer orden (cambio lineal en precio).
        gamma : float
            Sensibilidad de segundo orden (convexidad).
        """
        delta = self.pv_total_flows.dot(np.arange(1, self.pending_cupons + 1)) / 100
        gamma = self.pv_total_flows.dot(np.arange(1, self.pending_cupons + 1) **2) / 10000

        return delta, gamma
