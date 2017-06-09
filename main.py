


data = load_dataset('adult', 'train')

config = {}
config['num_hidden_layers'] = 3
config['weights_sharing'] = True

graph = tf.Graph()
dns = []
for i in range(max(data['targets'])+1):
    dn = DependencyNetwork(data['inputs_block'], data['attr_types'], graph=graph, graph_config=config, name="DN{0}".format(i+1))
    dns.append(dn)
sess = tf.InteractiveSession(graph=graph)
with graph.as_default():
    init = tf.global_variables_initializer()
sess.run(init)

for dn in dns:
    dn.set_session(sess)

train_inputs = data['inputs'][:2000, :]
train_targets = data['targets'][:2000]
valid_inputs = data['inputs'][2000:4000, :]
valid_targets = data['targets'][2000:4000]

for i, dn in enumerate(dns):
    dn.train(train_inputs[train_targets==i, :], valid_inputs[valid_targets==i, :], num_epoch=100, batch_size=100)

gen = SyntheticDataGenerator(train_inputs, train_targets, dns)
gen_inputs, gen_targets = gen.generate(num_round=1, include_original_data=False)

#sess.close()
