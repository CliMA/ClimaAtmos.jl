using MPI, CUDA
import Base: UUID

MPI.Init()

device_uuids = [
 UUID("09c6a428-f4ea-8730-fd1c-f3d323d0e161")
 UUID("02743130-9954-abbd-7a10-877b693748d6")
 UUID("393d0468-49c6-673b-766d-56b816b0710a")
 UUID("dc2578fd-6065-eca0-9ba6-dfdaef2ef24f")
 UUID("1768fcec-d945-7435-1f8e-85d30cdf310e")
 UUID("6420b6b9-bb34-a58d-8090-61887fd97931")
 UUID("9c342aaf-0c25-ccfb-5ac7-986af05735c3")
 UUID("b85b796f-6b3a-678c-ad94-20b94de6460b")
]

device_to_num = Dict(uuid => i-1 for (i,uuid) in enumerate(device_uuids))

comm_rank = MPI.Comm_rank(MPI.COMM_WORLD)
comm_size = MPI.Comm_size(MPI.COMM_WORLD)
CUDA.device!(comm_rank % CUDA.ndevices())
d = CUDA.device()

@info("info",
      comm_rank,
      device_uuid=uuid(d),
      device_num=device_to_num[uuid(d)]
      )

send_arr = CuArray{Float64}(undef, 1000)
fill!(send_arr, 1.0)
recv_arr = CuArray{Float64}(undef, 1000)

send_req = MPI.Isend(send_arr, MPI.COMM_WORLD; dest=(comm_rank + 1) % comm_size)
recv_req = MPI.Irecv!(recv_arr, MPI.COMM_WORLD; source=(comm_rank - 1)%comm_size)

MPI.Waitall([send_req,recv_req])
