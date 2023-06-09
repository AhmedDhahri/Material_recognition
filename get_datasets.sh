if test -d "downloads"; then
	echo "downloads exist."
else
	mkdir downloads
fi

cd downloads

if test -f "minc-model.tar.gz"; then
	echo "minc-model.tar.gz exist."
else
	wget http://opensurfaces.cs.cornell.edu/static/minc/minc-model.tar.gz
	tar -xvf minc-model.tar.gz
	mv minc-model/* ../datasets/models
fi

if test -f "minc-original-photo.tar.gz"; then
	echo "minc-original-photo.tar.gz exist."
else
	wget http://opensurfaces.cs.cornell.edu/static/minc/minc-original-photo.tar.gz
	tar -xvf minc-original-photo.tar.gz --directory ../datasets/
	mv ../datasets/photo_prig ../datasets/minc
fi

if test -f "minc.tar.gz"; then
	echo "minc.tar.gz exist."
else
	wget http://opensurfaces.cs.cornell.edu/static/minc/minc.tar.gz
	tar -xvf minc.tar.gz 
	mv minc/* ../datasets/minc
fi

if test -f "Irh_dataset.zip"; then
	echo "Irh_dataset.zip exist."
else
	wget https://github.com/AhmedDhahri/Material_recognition/releases/download/datasets/Irh_dataset.zip
	mkdir ../datasets/irh
	mkdir datasets/irh/files
	unzip Irh_dataset.zip -d datasets/irh/files
fi

