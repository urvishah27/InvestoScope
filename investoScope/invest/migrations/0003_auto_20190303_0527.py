# Generated by Django 2.1.7 on 2019-03-02 23:57

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('invest', '0002_auto_20190303_0516'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='purchase',
            name='type',
        ),
        migrations.AddField(
            model_name='purchase',
            name='stock_type',
            field=models.CharField(choices=[('Forex', 'Forex'), ('Stock', 'Stock'), ('Crypto', 'Crypto')], default='Stock', max_length=20),
        ),
    ]
