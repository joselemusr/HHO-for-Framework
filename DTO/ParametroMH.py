class ParametroMH:
    def __init__(self):
        self.nombre = None
        self.valor = None
        print(f"ParametroMH creado")

    def setNombre(self, nombre):
        self.nombre = nombre

    def getNombre(self):
        return self.nombre
    
    def setValor(self, valor):
        self.valor = valor

    def getValor(self):
        return self.valor