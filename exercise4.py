# --------------------------------------------------------------------------------------------------------------------------
# Ana de la Cerda Galvan ------- Exercise 4 - NHL REQUERIMENTS  ------------------------------------------------------------
# There is some Spanish code inside, I introduced them while I studied the code to be clear of what id does en every step
#---------------------------------------------------------------------------------------------------------------------------
# STRUCTURE BUG
# In our code we have an structure bag relationated with using a fixed size of the banch.
# Because when we change the batch_size from 32 to 64, the last mini-batch could contain fewer than 64 images.
# However, the code still created 64 fixed labels (torch.ones((batch_size, 1))), causing a size mismatch between images and labels.
# COSMETIC BUG
# Here the problem whas easier, we only have to control de margen of the imagens after the subtitles not before to avoid overlapping.
#---------------------------------------------------------------------------------------------------------------------------------


def train_gan(batch_size: int = 32, num_epochs: int = 100, device: str = "cuda:0" if torch.cuda.is_available() else "cpu"):
      

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    # Compose:encadena varias transformacions. ToTensor:convierte la imagen a un tensor Pytorch. Normalize: [0,1]--> [-1,1]
    # Trabajamos mejor con datos centrados en 0, nedia =0;
    
    try:
        train_set = torchvision.datasets.MNIST(root=".", train=True, download=True, transform=transform)
        # Descargamos la base de datos MNIST en . = directorio actual.Train (True=60000 imag/False=10000). Vuelve a hacer la transformacion anterior.
        # Cada vez que cargue una imag estará ya procesada como queremos.
    except:
        print("Failed to download MNIST, retrying with different URL")
        # see: https://github.com/pytorch/vision/blob/master/torchvision/datasets/mnist.py
        torchvision.datasets.MNIST.resources = [
            ('https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz',
             'f68b3c2dcbeaaa9fbdd348bbdeb94873'),
            ('https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz',
             'd53e105ee54ea40749a09fcbcd1e9432'),
            ('https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz',
             '9fb629c4189551a2d022fa330f9573f3'),
            ('https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz',
             'ec29112dd5afa0611ce80d1b7f02629c')
            #Cambia las url de donde intenta bajar la base de datos
        ]
        train_set = torchvision.datasets.MNIST(root=".", train=True, download=True, transform=transform)
        #vuelve a descargarla con todas las opciones que vimos antes

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
    

    # Set up training
    discriminator = Discriminator().to(device)
    generator = Generator().to(device)
    #creamos dos objetos de ambas clases que correran segun device(GPU=+rapido/CPU=mas lento)
    lr = 0.0001
    #lr=taza de aprendizaje=muy baja pero mas estable. Si fuera alta >=0.1
    loss_function = nn.BCELoss()
    #mide que bien el discriminador diferencia entre real y falso
   
    optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=lr)
    optimizer_generator = torch.optim.Adam(generator.parameters(), lr=lr)
    #Usamos el optimizador Adam para ajustar los pesos de la red neuronal para que la pérdida (loss) vaya disminuyendo.

    # train
    for epoch in range(num_epochs):
        #hacemos num_epochs, numero de veces que vamos a ver todas las imagenes que estan en el segundo buble.
        for n, (real_samples, mnist_labels) in enumerate(train_loader):

            # Data for training the discriminator
            # Coge un batch de imágenes reales y les asigna etiqueta 1.
            # Genera imágenes falsas a partir de ruido gaussiano usando el generador, y les asigna etiqueta 0.
            # Las une (reales + falsas) en un solo conjunto con sus etiquetas correspondientes.
            
            real_samples = real_samples.to(device=device)
            
            # BUG
            # When batch_size is set to 64, the last batch may have fewer samples, but the code still creates 64 labels. 
            # This causes a size mismatch between images and labels, leading to an error.
            # Here we are going to change the batch_size, instead of using a fixed one I'm going to use the current mini-batch
            # Then when we change from the 32 to 64 the last  batch which smaller size wouldn't have any problem. (current_bs)

            actual_bs= real_samples.size(0)
            real_samples_labels = torch.ones((actual_bs, 1)).to(device=device)
            latent_space_samples = torch.randn((actual_bs, 100)).to(device=device)
            generated_samples = generator(latent_space_samples)
            generated_samples_labels = torch.zeros((actual_bs, 1)).to(device=device)
        
            all_samples = torch.cat((real_samples, generated_samples))
            all_samples_labels = torch.cat((real_samples_labels, generated_samples_labels))

            
            # Training the discriminator
            discriminator.zero_grad()
            output_discriminator = discriminator(all_samples)
            loss_discriminator = loss_function(output_discriminator, all_samples_labels)
            loss_discriminator.backward()
            optimizer_discriminator.step()

            # Data for training the generator
            latent_space_samples = torch.randn((batch_size, 100)).to(device=device)

            # Training the generator
            generator.zero_grad()
            generated_samples = generator(latent_space_samples)
            output_discriminator_generated = discriminator(generated_samples)
            loss_generator = loss_function(output_discriminator_generated, real_samples_labels)
            loss_generator.backward()
            optimizer_generator.step()

            if n == batch_size - 1:
                name = f"Generate images\n Epoch: {epoch} Loss D.: {loss_discriminator:.2f} Loss G.: {loss_generator:.2f}"
                generated_samples = generated_samples.detach().cpu().numpy()
                fig = plt.figure()
                for i in range(16):
                    sub = fig.add_subplot(4, 4, 1 + i)
                    sub.imshow(generated_samples[i].reshape(28, 28), cmap="gray_r")
                    sub.axis('off')
                
                # Here we have the cosmetic bug, we adjunst the margins after the subtitle, so it won't be overloap for the imagens
                fig.suptitle(name)
                fig.tight_layout()
                clear_output(wait=False)
                display(fig)


        print ("Training finished.")
        
train_gan(batch_size=32, num_epochs=100)
