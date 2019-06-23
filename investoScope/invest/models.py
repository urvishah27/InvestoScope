from django.db import models
from django.contrib.auth.models import User
# Create your models here.

class Purchase(models.Model):
    id = models.AutoField(primary_key=True)
    stock = models.CharField(max_length=20, default='BA')
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    units = models.FloatField(default=0)
    per_unit_price = models.FloatField(default=0)
    TYPE_CHOICES = (('Forex','Forex'), ('Stock','Stock'), ('Crypto','Crypto'))
    def get_investment(this):
        return float(this.units * this.per_unit_price)
    stock_type = models.CharField(max_length=20,choices=TYPE_CHOICES, default='Stock')