# Generated by Django 2.1.7 on 2019-03-02 23:58

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('invest', '0003_auto_20190303_0527'),
    ]

    operations = [
        migrations.AlterField(
            model_name='purchase',
            name='units',
            field=models.FloatField(default=0),
        ),
    ]
