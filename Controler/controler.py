import keyboard
import threading
import XInput

##Const estados botones
# 0 es el boton suelto
# 1 es el boton que se aprieta y se suelta
# 2 es el boton apretado indefinidamente
#Booleano si esta activo o no
OffState=0
PnRState=1
PressState=2

activo=True
#Los botones del SNES son los siguientes
#a, b, x, y, r, l, start, select, arriba, abajo, izquierda, derecha
#este script accepta botones con estados, los botones comienzan con el estado off por defecto, todos los botones pueden ser apretados al tiempo,
#con excepcion de arriba y abajo al tiempo, e izquierda y derecha al tiempo
#los estados de los botones se manejan en un diccionario
global botList
botList= {'a':0, 'b':0, 'x':0, 'y':0, 'r':0, 'l':0, 'up':0, 'down':0, 'left':0, 'right':0}

def getState():
    return botList

def changeState(boton, state):
    if(state==OffState):
        if(botList[boton]==OffState):
            pass
        else:
            botList[boton]=OffState

    if(state==PnRState):
            if(botList[boton]==PnRState):
                pass
            else:
                botList[boton]=PnRState
    if(state==PressState):
            if(botList[boton]==PressState):
                pass
            else:
                botList[boton]=PressState
    else:
        print("Invalid State")


def buttonPresser():
    while activo:
        for a in botList:
            if(a==PnRState):
                keyboard.press_and_release(boton)
                botList[a]=OffState
            while(a==PressState):
                keyboard.press(boton)

def main():
    threads = []
    threads.append(threading.Thread(target=buttonPresser))
    threads[0].start()

if __name__ == "__main__":
    main()
