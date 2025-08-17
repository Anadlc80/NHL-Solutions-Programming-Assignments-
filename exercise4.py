train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    # Train_loader vble que va guardando los grupos de imagenes; Dataloader: organiza los datos en grupos: bath_size=tamaño del grupo; shuffer=aleatorio




    
    # example data
    real_samples, mnist_labels = next(iter(train_loader))
    #creamos untinerador que recorre el primer grupo. Par de imagen/etiqueta. Real_sample(num de imagenes en el grupo,canal(1(blanco/negro),tamaño)
    #mnist_label=nos indica que dígito es la imagen correspondiente. Cada imagen lleva una etiqueta que dice el número que es.

    fig = plt.figure()
    #creamos un lienzo donde podemos dibujar
    for i in range(16):
        sub = fig.add_subplot(4, 4, 1 + i)
        sub.imshow(real_samples[i].reshape(28, 28), cmap="gray_r")
        sub.axis('off')
        # mostramos 16 imagnenes, cada una en una cuadricula de 4x4. Axis= quita los ejes para ver las imagenes claras

    
    fig.suptitle("Real images")
    fig.tight_layout()
    display(fig)
    #mostramos esas imagenes con un titulo que indica que son las verdades, controlamos los margenes (tight_layaout()
    
    time.sleep(5)
    #paramos la ejecucion durante 5 segundos
