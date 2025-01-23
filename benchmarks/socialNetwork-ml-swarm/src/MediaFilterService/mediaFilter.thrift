namespace py social_network

service MediaFilterService {
    list<bool> UploadMedia(1: i32 req_id, 2: list<string> media_types, 3: list<string> medium, 4: string carrier)
}
