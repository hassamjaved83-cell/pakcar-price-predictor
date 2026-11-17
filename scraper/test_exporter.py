from exporter import Exporter

cars = [
    {
        "title": "Suzuki Alto",
        "price": "2500000",
        "ad_ref": "12345"
    },
    {
        "title": "Honda City",
        "price": "4300000",
        "ad_ref": "54321"
    }
]

exporter = Exporter()

exporter.save(cars)