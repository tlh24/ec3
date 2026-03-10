module BSQ = struct
	(* Bounded Set Queue *)
	(* A node in our doubly linked list *)
	type node = {
		value : string;
		mutable prev : node option;
		mutable next : node option;
	}

	(* The main data structure *)
	type t = {
		capacity : int;
		table : (string, node) Hashtbl.t;
		mutable head : node option; (* Front of queue (Most recently added) *)
		mutable tail : node option; (* Back of queue (Oldest) *)
	}

	(* Initialize the structure with a maximum capacity *)
	let create capacity =
		if capacity <= 0 then invalid_arg "Capacity must be > 0";
		{
		capacity;
		table = Hashtbl.create capacity;
		head = None;
		tail = None;
		}

	(* Internal Helper: Plucks a node out of its current place in the list *)
	let remove_node t node =
		let p = node.prev in
		let n = node.next in
		(match p with
		| Some prev_node -> prev_node.next <- n
		| None -> t.head <- n);
		(match n with
		| Some next_node -> next_node.prev <- p
		| None -> t.tail <- p);
		node.prev <- None;
		node.next <- None

	(* Internal Helper: Inserts a node at the very front of the list *)
	let add_to_head t node =
		node.next <- t.head;
		node.prev <- None;
		(match t.head with
		| Some head_node -> head_node.prev <- Some node
		| None -> t.tail <- Some node);
		t.head <- Some node

	(* Main Operation: Add a string *)
	let add t v =
		match Hashtbl.find_opt t.table v with
		| Some node ->
			(* Case 1: String is already in the set.
				Reset its priority by moving it to the front of the queue. *)
			remove_node t node;
			add_to_head t node
		| None ->
			(* Case 2: New string. *)
			let node = { value = v; prev = None; next = None } in

			(* If we are at capacity, drop the lowest priority (tail) string *)
			if Hashtbl.length t.table >= t.capacity then (
				match t.tail with
				| Some tail_node ->
					Hashtbl.remove t.table tail_node.value;
					remove_node t tail_node
				| None -> ()
			);

			(* Add the new string to the hash table and the front of the list *)
			Hashtbl.add t.table v node;
			add_to_head t node

	(* Check if a string is currently in the set: O(1) *)
	let mem t v =
		Hashtbl.mem t.table v
	let length t =
		Hashtbl.length t.table

	(* O(N) array-like access (0 is the newest/head, length-1 is the oldest) *)
	let get t i =
		if i < 0 || i >= length t then invalid_arg "Index out of bounds";
		let rec find node idx =
		if idx = 0 then node.value
		else match node.next with
			| Some n -> find n (idx - 1)
			| None -> failwith "Unreachable"
		in
		match t.head with
		| Some head_node -> find head_node i
		| None -> failwith "Unreachable"

	(* Saves to TXT file from oldest to newest *)
	let save_to_file t filename =
		let oc = open_out filename in
		(* Traverse backwards using `.prev` starting from the tail *)
		let rec write_oldest_to_newest = function
		| None -> ()
		| Some node ->
			output_string oc (node.value ^ "\n");
			write_oldest_to_newest node.prev
		in
		write_oldest_to_newest t.tail;
		close_out oc

	(* Loads from TXT file, adding top-to-bottom *)
	let load_from_file capacity filename =
		let t = create capacity in
		let ic = open_in filename in
		try
		let rec loop () =
			let line = input_line ic in
			add t line;  (* Re-adds items, naturally pushing newest to the front *)
			loop ()
		in
		loop ()
		with End_of_file ->
		close_in ic;
		t

	(* Utility to view the current queue from newest to oldest *)
	let to_list t =
		let rec aux acc = function
		| None -> List.rev acc
		| Some node -> aux (node.value :: acc) node.next
		in
		aux[] t.head
end
