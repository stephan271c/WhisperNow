use std::{
    io::{ErrorKind, Read, Write},
    net::{TcpListener, TcpStream},
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};

const DEADLINE: Duration = Duration::from_secs(5);

pub struct Received {
    pub head: String,
    pub body: Vec<u8>,
}

pub fn serve_once(status_and_headers: &str, body: &str) -> (String, JoinHandle<Received>) {
    let listener = TcpListener::bind("127.0.0.1:0").expect("fixture listener");
    listener
        .set_nonblocking(true)
        .expect("nonblocking fixture listener");
    let address = format!("http://{}", listener.local_addr().expect("fixture address"));
    let response = format!(
        "HTTP/1.1 {status_and_headers}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    );
    let server = thread::spawn(move || {
        let mut connection = accept(&listener);
        let received = read_request(&mut connection);
        connection
            .write_all(response.as_bytes())
            .expect("fixture response");
        received
    });
    (address, server)
}

fn accept(listener: &TcpListener) -> TcpStream {
    let deadline = Instant::now() + DEADLINE;
    loop {
        match listener.accept() {
            Ok((connection, _)) => {
                connection
                    .set_nonblocking(false)
                    .expect("blocking fixture connection");
                connection
                    .set_read_timeout(Some(DEADLINE))
                    .expect("request timeout");
                return connection;
            }
            Err(error) if error.kind() == ErrorKind::WouldBlock => {
                assert!(
                    Instant::now() < deadline,
                    "the provider did not connect before the fixture deadline"
                );
                thread::sleep(Duration::from_millis(10));
            }
            Err(error) if error.kind() == ErrorKind::Interrupted => {}
            Err(error) => panic!("accept provider request: {error}"),
        }
    }
}

fn read_request(connection: &mut TcpStream) -> Received {
    let mut request = Vec::new();
    let mut buffer = [0_u8; 4096];
    let head_length = loop {
        if let Some(end) = request.windows(4).position(|window| window == b"\r\n\r\n") {
            break end + 4;
        }
        let read = connection.read(&mut buffer).expect("request bytes");
        assert!(read > 0, "request ended before its headers");
        request.extend_from_slice(&buffer[..read]);
    };
    let head = String::from_utf8(request[..head_length].to_vec()).expect("request head");
    let body_length = head
        .lines()
        .find_map(|line| {
            let (name, value) = line.split_once(':')?;
            name.eq_ignore_ascii_case("content-length")
                .then(|| value.trim().parse::<usize>().expect("body length"))
        })
        .unwrap_or(0);
    while request.len() < head_length + body_length {
        let read = connection.read(&mut buffer).expect("request body bytes");
        assert!(read > 0, "request ended before its body");
        request.extend_from_slice(&buffer[..read]);
    }
    Received {
        head,
        body: request[head_length..head_length + body_length].to_vec(),
    }
}
