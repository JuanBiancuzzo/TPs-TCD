from typing import List, Dict, Self
from enum import Enum

class LadoArbol(Enum):
    IZQUIERDA = False
    DERECHA = True

    def valor(self) -> bool:
        return self == LadoArbol.DERECHA

class Nodo:
    def __init__(self):
        self.padre = None
        self.lado = None

        self.izquierda = None
        self.derecha = None

    def Hoja(caracter: str, apariciones: int) -> Self:
        self = Nodo()

        self.caracter = caracter
        self.apariciones = apariciones

        return self

    def Nodo(izquierda: Self, derecha: Self) -> Self:
        self = Nodo()

        izquierda.asignarPadre(self, LadoArbol.IZQUIERDA)
        derecha.asignarPadre(self, LadoArbol.DERECHA)

        self.izquierda = izquierda
        self.derecha = derecha
        self.apariciones = izquierda.apariciones + derecha.apariciones

        return self

    def asignarPadre(self, padre: Self, lado: LadoArbol):
        self.padre = padre
        self.lado = lado

    def representacion(self) -> List[bool]:
        if self.padre is None or self.lado is None:
            return []

        representacion = self.padre.representacion()
        representacion.append(self.lado.valor())
        return representacion

    def descodificar(self, secuencia: List[bool]) -> str:
        if self.soyHoja():
            return self.caracter

        if secuencia.pop(0):
            return self.derecha.descodificar(secuencia)
        else: 
            return self.izquierda.descodificar(secuencia)
    
    def soyHoja(self):
        return self.izquierda == None or self.derecha == None

    def __str__(self): 
        if self.soyHoja():
            return f"({self.caracter}: {self.apariciones})"

        repIzquierdo = "\n\t".join(self.izquierda.__str__().split("\n"))
        repDerecho = "\n\t".join(self.derecha.__str__().split("\n"))

        return f"Nodo con {self.apariciones}:\n\t - {repIzquierdo}\n\t - {repDerecho}"

def insersionOrdenada(elementos: List[Nodo], elemento: Nodo):
    posicion = 0
    while posicion < len(elementos):
        # Ver si es mejor < o <=, uno de los dos nos va a dejar tener codigos más parejos, aka varianza mas chica
        if elemento.apariciones <= elementos[posicion].apariciones:
            break

        posicion += 1

    elementos.insert(posicion, elemento)

class Huffman:
    def __init__(self, apariciones: Dict[str, int]):
        caracteres: Dict[str, Nodo] = {}

        # Crear un heap maximal con apariciones
        lista: List[Nodo] = []
        for caracter, aparicion in apariciones.items():
            nuevoCaracter = Nodo.Hoja(caracter, aparicion)
            caracteres[caracter] = nuevoCaracter
            insersionOrdenada(lista, nuevoCaracter)

        # Verificar si el heap tiene elementos
        while len(lista) > 1:
            # Sacamos dos, creamos un nodo 
            primero, segundo, lista = *lista[:2], lista[2:] 
            self.nodoPrincipal = Nodo.Nodo(primero, segundo)

            #  * Si tiene al menos un elemento => insertar el nodo, y repetir
            insersionOrdenada(lista, self.nodoPrincipal)

        self.caracteres: Dict[str, List[bool]] = {}
        for caracter, hoja in caracteres.items():
            self.caracteres[caracter] = hoja.representacion() 

    def encodear(self, texto: List[str]) -> List[bool]:
        resultado: List[bool] = []
        for caracter in texto:
            resultado.extend(self.caracteres[caracter])
        return resultado

    def descodificar(self, secuencia: List[bool]) -> List[str]: 
        texto = []
        while len(secuencia) > 0:
            texto.append(self.nodoPrincipal.descodificar(secuencia))
        return texto

huff = Huffman({
    "A": 2,
    "D": 5,
    "B": 3,
    "E": 1,
})

texto = "ADBADEDBBDD"
codigo = huff.encodear(texto)
destexto = "".join("1" if bit else "0" for bit in codigo)

print( "Deberia ser: 10101110101000111100")
print(f"y es:        {destexto}")

print("")

desdestexto = "".join(huff.descodificar(codigo))
print(f"Deberia ser: {texto}")
print(f"y es:        {desdestexto}")