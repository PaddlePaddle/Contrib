                # x_src = to_variable(np.load('/home/j/Desktop/stargan-v2-master/save_data/x_real_{}.npy'.format('test')))
                # x_ref = to_variable(np.load('/home/j/Desktop/stargan-v2-master/save_data/x_ref_{}.npy'.format('test')))
                # y_trg = to_variable(np.load('/home/j/Desktop/stargan-v2-master/save_data/y_ref_{}.npy'.format('test')))
                #
                # s_ref = self.style_encoder([x_ref, y_trg])
                # s_ref = s_ref.numpy()
                # torch_s_ref = to_variable(np.load('/home/j/Desktop/stargan-v2-master/save_data/s_ref_{}.npy'.format('test')))
                #
                # print(s_ref)
                # print('xxxxxxxxxxxxxxx')
                # print(torch_s_ref.numpy())

                z_many = to_variable(np.load('/home/j/Desktop/stargan-v2-master/save_data/z_trg_{}.npy'.format('1')))
                y_many = to_variable(np.load('/home/j/Desktop/stargan-v2-master/save_data/y_trg_{}.npy'.format('1')))

                latent_fake = to_variable(
                    np.load('/home/j/Desktop/stargan-v2-master/save_data/latent_fake_{}.npy'.format('1')))

                s_ref = self.mapping_network([z_many, y_many])

                torch_s_ref = to_variable(
                    np.load('/home/j/Desktop/stargan-v2-master/save_data/s_trg_{}.npy'.format('1')))

                x_src = to_variable(np.load('/home/j/Desktop/stargan-v2-master/save_data/x_src_{}.npy'.format('1')))
                fake = self.generator([x_src, s_ref])
                # #print(fake.numpy())
                # diff = abs(s_ref - torch_s_ref.numpy())
                # diff = abs(fake.numpy() - latent_fake.numpy())
                #
                torch_fake = np.load('/home/j/Desktop/stargan-v2-master/save_data/latent_fake_{}.npy'.format('1'))
                print('fake.shape',fake.shape)
                diff = abs(fake.numpy() - torch_fake)
                sss=abs(np.sum(fake.numpy()) - np.sum(torch_fake))
                print('ssssssss',sss)
                diffs = np.sum(diff)
                print('diffs', diffs)
                print('diff:', diff)
                # print('pdd:{},ptd:{},diff:{}'.format( float(s_ref), float(torch_s_ref), float(diff)))
                if float(diff) < 1e-6:
                    print('True')
                else:
                    print('False')
