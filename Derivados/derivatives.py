"""
Descripción:   Clases para la valuacion de derivados
Autor:         David Jiménez Cooper - SpiderCoop
Fecha:         2026-06-14
"""

# Importamos Librerías
import numpy as np
import pandas as pd
import math


# Definimos la clase forward
class Forward:
    def __init__(self, position:str, expire_date:pd.Timestamp, emition_date:pd.Timestamp, calendar_convention:str = 'Actual/360'):
        
        # Convertimos los parametros a tipo correspondiente
        self.position = position.strip().lower()
        assert self.position in ('long','short'), 'Position not well defined'

        # Convertimos las fechas a tipo datetime
        self.expire_date = pd.to_datetime(expire_date)
        self.emition_date = pd.to_datetime(emition_date)

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

        assert start_date < end_date, 'end date must be after start date'

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
    def _ensure_decimal_rate(rate):
        """
        Convierte tasas a decimales. Soporta float, list, numpy array o pandas Series.
        """
        if isinstance(rate, pd.Series):
            return rate.apply(lambda x: x/100 if x > 1 else x)
        elif isinstance(rate, (list, np.ndarray)):
            return pd.Series(rate).apply(lambda x: x/100 if x > 1 else x).values
        elif isinstance(rate, (int, float)):
            return rate/100 if rate > 1 else rate
        else:
            raise TypeError("Rate must be float, int, list, numpy array, or pandas Series")


    
    def compute_forward_price(self, spot_price:float, risk_free_rate:float, valuation_date:pd.Timestamp, pv_income:float = 0, yield_rate:float = 0):
        
        # Verificamos que las tasas sean correctas
        risk_free_rate = self._ensure_decimal_rate(risk_free_rate)
        yield_rate = self._ensure_decimal_rate(yield_rate)

        # Obtenemos el plazo a expiracion de acuerdo a la convencion del calendario
        days_to_expire, days_to_expire_as_year_fraction = self.day_count_fraction(valuation_date, self.expire_date)

        # Obtenemos la tasa de descuento, en caso de que yield_rate sea distinto de cero se trata de subyacente con dividendos, con retorno por conveniencia o de divisa
        risk_free_rate = risk_free_rate[days_to_expire] if isinstance(risk_free_rate, pd.Series) else risk_free_rate
        yield_rate = yield_rate[days_to_expire] if isinstance(yield_rate, pd.Series) else yield_rate

        # En caso de que yield_rate sea distinto de cero se trata de subyacente con dividendos, con retorno por conveniencia o de divisa
        forward_price = (spot_price - pv_income) * np.exp((risk_free_rate - yield_rate)*days_to_expire_as_year_fraction)

        return forward_price



    def valuate(self, spot_price:float, risk_free_rate:float|pd.Series, K:float = 0, valuation_date:pd.Timestamp = pd.Timestamp.today(), pv_income:float = 0, yield_rate:float|pd.Series = 0):
        
        # Obtenemos las variables 
        self.spot_price = spot_price
        self.risk_free_rate = self._ensure_decimal_rate(risk_free_rate)
        self.yield_rate = self._ensure_decimal_rate(yield_rate)
        self.valuation_date = pd.to_datetime(valuation_date)

        # Obtenemos el plazo a expiracion de acuerdo a la convencion del calendario
        self.days_to_expire, self.days_to_expire_as_year_fraction = self.day_count_fraction(self.valuation_date, self.expire_date)

        # Obtenemos la tasa de descuento, en caso de que yield_rate sea distinto de cero se trata de subyacente con dividendos, con retorno por conveniencia o de divisa
        risk_free_rate = self.risk_free_rate[self.days_to_expire] if isinstance(self.risk_free_rate, pd.Series) else self.risk_free_rate
        yield_rate = self.yield_rate[self.days_to_expire] if isinstance(self.yield_rate, pd.Series) else self.yield_rate

        # Obtenemos el precio forward
        self.forward_price = self.compute_forward_price(self.spot_price, risk_free_rate, self.valuation_date, pv_income, yield_rate)

        # Calculamos la segunda parte de la valuacion del forward
        vp_forward = K * np.exp(-risk_free_rate*self.days_to_expire_as_year_fraction)
        vp_price = (self.spot_price - pv_income) * np.exp(-yield_rate*self.days_to_expire_as_year_fraction)

        if self.position=='long':
            self.valuation = vp_price - vp_forward
        elif self.position=='short':
            self.valuation = vp_forward - vp_price

        return self.valuation
    