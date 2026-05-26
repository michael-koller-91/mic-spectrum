main: main.odin
	odin run .

release: main.odin
	odin build . -o:speed

download-odin:
	mkdir downloads
	# download and unpack Odin
	wget https://github.com/odin-lang/Odin/releases/download/dev-2026-03/odin-linux-amd64-dev-2026-03.tar.gz
	tar -xf odin-linux-amd64-dev-2026-03.tar.gz
	rm odin-linux-amd64-dev-2026-03.tar.gz
	# move to downloads
	mv odin-linux-amd64-nightly+2026-03-03 downloads
